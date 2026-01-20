from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from enum import Enum
from hashlib import md5
from pathlib import Path
from platform import system
from shutil import rmtree
from tempfile import gettempprefix, mkdtemp
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from warnings import warn

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from df_analyze.analysis.univariate.associate import AssocResults
    from df_analyze.analysis.univariate.predict.predict import PredResults
    from df_analyze.hypertune import EvaluationResults
    from df_analyze.preprocessing.cleaning import RenameInfo
    from df_analyze.preprocessing.inspection.inspection import InspectionResults
    from df_analyze.preprocessing.prepare import PreparedData
    from df_analyze.selection.filter import FilterSelected
    from df_analyze.selection.models import ModelSelected
from df_analyze.utils import Debug

JOBLIB = "__JOBLIB_CACHE__"


class FileType(Enum):
    Interim = "interim"
    Final = "final"
    Feature = "feature"
    Params = "params"
    Univariate = "univariate"


@dataclass
class ProgramDirs(Debug):
    """Container for various output and caching directories"""

    root: Optional[Path] = None
    terminal: Optional[Path] = None
    options: Optional[Path] = None
    joblib_cache: Optional[Path] = None
    renames: Optional[Path] = None
    inspection: Optional[Path] = None
    prepared: Optional[Path] = None
    features: Optional[Path] = None
    descriptions: Optional[Path] = None
    associations: Optional[Path] = None
    predictions: Optional[Path] = None
    selection: Optional[Path] = None
    filter: Optional[Path] = None
    embed: Optional[Path] = None
    wrapper: Optional[Path] = None
    tuning: Optional[Path] = None
    results: Optional[Path] = None
    needs_clean: bool = False

    @staticmethod
    def new(root: Optional[Path], hsh: str) -> ProgramDirs:
        root, needs_clean = ProgramDirs.configure_root(root)

        if root is None:
            return ProgramDirs()
        root = root / hsh

        new = ProgramDirs(
            root=root,
            options=root / "options.json",
            terminal=root / "terminal_outputs.txt",
            joblib_cache=root / "__JOBLIB_CACHE__",
            renames=root / "feature_renamings.md",
            inspection=root / "inspection",
            prepared=root / "prepared",
            features=root / "features",
            descriptions=root / "features/descriptions",
            associations=root / "features/associations",
            predictions=root / "features/predictions",
            selection=root / "selection",
            filter=root / "selection/filter",
            embed=root / "selection/embed",
            wrapper=root / "selection/wrapper",
            tuning=root / "tuning",
            results=root / "results",
            needs_clean=needs_clean,
        )
        for attr, path in new.__dict__.items():
            if isinstance(path, Path):
                if "JOBLIB" in str(path):  # joblib handles creation
                    continue
                if attr in ["options", "renames"]:
                    continue
                if attr == "terminal":
                    path.touch()
                    continue
                try:
                    path.mkdir(exist_ok=True, parents=True)
                except Exception:
                    warn(
                        "Got error creating output directories:\n"
                        f"{traceback.format_exc()}.\n"
                        "Defaulting to in-memory results."
                    )
                    return ProgramDirs(root=None)
        return new

    @staticmethod
    def configure_root(root: Optional[Path] = None) -> tuple[Optional[Path], bool]:
        try:
            if (root is not None) and (not os.access(root, os.W_OK)):
                raise PermissionError(
                    f"Provided outdir: {root} is not writeable. Provide a "
                    "different directory or pass `None` to allow `df-analyze` "
                    "to configure this automatically."
                )

            needs_clean = False
            outdir = root or Path.home().resolve() / "df-analyze-outputs"
            if system().lower() == "windows":
                return outdir, needs_clean

            if not os.access(outdir, os.W_OK):
                new = Path.cwd().resolve() / "df-analyze-outputs"
                warn(f"Do not have write permissions for {outdir}. Trying: {new}.")
                outdir = new
            else:
                outdir.mkdir(exist_ok=True, parents=True)
                return outdir, needs_clean

            if not os.access(outdir, os.W_OK):
                try:
                    raw = mkdtemp(prefix="df_analyze_tmp")
                    needs_clean = True
                    new = Path(raw)
                    warn(
                        f"Did not have write permissions for {outdir}. Created a "
                        f"temporary directory at {new} instead."
                    )
                    outdir = new
                    outdir.mkdir(exist_ok=True, parents=True)
                    return outdir, needs_clean
                except FileExistsError:
                    warn(
                        f"Found existing temporary data. Likely this exists "
                        f"at {Path(gettempprefix()).resolve()} and must be "
                        "deleted. Holding results in memory instead. "
                    )
                    return None, needs_clean
                except Exception as e:
                    warn(
                        "Could not create temporary directory. Falling back to "
                        f"storing results in memory. Error details:\n{e}"
                        f"{traceback.format_exc()}"
                    )
                    return None, needs_clean

            else:
                outdir.mkdir(exist_ok=True, parents=True)
                return outdir, needs_clean
        except Exception as e:
            warn(
                "Could not create temporary directory. Falling back to "
                f"storing results in memory. Error details:\n{e}"
                f"{traceback.format_exc()}"
            )
            return None, False

    def save_renames(self, renames: RenameInfo) -> None:
        if self.renames is None:
            return
        out = self.renames
        try:
            report = renames.to_markdown()
            if report is None:
                return
            out.write_text(report, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save feature renames. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_eval_report(
        self, results: Optional[EvaluationResults], fold_idx: Optional[int] = None
    ) -> None:
        if self.results is None:
            return
        if results is None:
            return
        out = add_fold_idx(self.results / "results_report.md", fold_idx=fold_idx)
        try:
            out.write_text(results.to_markdown(), encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save final evaluation report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_eval_tables(
        self, results: Optional[EvaluationResults], fold_idx: Optional[int] = None
    ) -> None:
        if self.results is None or self.tuning is None:
            return
        if results is None:
            return
        perfs = add_fold_idx(self.results / "final_performances.csv", fold_idx=fold_idx)
        tuned = add_fold_idx(self.tuning / "tuned_models.csv", fold_idx=fold_idx)
        df_tuned = results.hp_table(fold_idx)
        try:
            results_df = results.df
            if fold_idx is not None:
                results_df["test_idx"] = fold_idx
            results_df.to_csv(perfs)
            df_tuned.to_csv(tuned)
        except Exception as e:
            warn(
                "Got exception when attempting to save final evaluation report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_eval_data(
        self, results: Optional[EvaluationResults], fold_idx: Optional[int] = None
    ) -> None:
        if self.results is None:
            return
        if results is None:
            return
        try:
            results.save(self.results, fold_idx=fold_idx)
        except Exception as e:
            warn(
                "Got exception when attempting to save final evaluation results. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_embed_report(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        if selected is None:
            return
        embeds = selected.embed_selected
        if embeds is None or (len(embeds) == 0):
            return
        if self.embed is None:
            return

        for embed_selected in embeds:
            report = embed_selected.to_markdown()
            model = embed_selected.model.value
            out = add_fold_idx(
                self.embed / f"{model}_embedded_selection_report.md", fold_idx=fold_idx
            )
            try:
                out.write_text(report, encoding="utf-8")
            except Exception as e:
                warn(
                    f"Got exception when attempting to save embedded selection report for model '{model}'. "
                    f"Details:\n{e}\n{traceback.format_exc()}"
                )

    def save_wrap_report(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        if selected is None:
            return
        if selected.wrap_selected is None:
            return
        if self.wrapper is None:
            return

        report = selected.wrap_selected.to_markdown()
        out = self.wrapper / "wrapper_selection_report.md"
        try:
            out.write_text(report, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save wrapper selection report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_wrap_data(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        if selected is None:
            return
        if selected.wrap_selected is None:
            return
        if self.wrapper is None:
            return

        json = selected.wrap_selected.to_json()
        out = self.wrapper / "wrapper_selection_data.json"
        try:
            out.write_text(json, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save wrapper selection data. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_embed_data(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        if selected is None:
            return
        embeds = selected.embed_selected
        if embeds is None or (len(embeds) == 0):
            return
        if self.embed is None:
            return

        for embed_selected in embeds:
            json = embed_selected.to_json()
            model = embed_selected.model.value
            out = add_fold_idx(
                self.embed / f"{model}_embed_selection_data.json", fold_idx
            )
            try:
                out.write_text(json, encoding="utf-8")
            except Exception as e:
                warn(
                    "Got exception when attempting to save embeded selection data. "
                    f"Details:\n{e}\n{traceback.format_exc()}"
                )

    def save_model_selection_reports(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        self.save_embed_report(selected, fold_idx=fold_idx)
        self.save_wrap_report(selected, fold_idx=fold_idx)

    def save_model_selection_data(
        self, selected: Optional[ModelSelected], fold_idx: Optional[int] = None
    ) -> None:
        self.save_embed_data(selected, fold_idx=fold_idx)
        self.save_wrap_data(selected, fold_idx=fold_idx)

    def save_filter_report(
        self, selected: Optional[FilterSelected], fold_idx: Optional[int] = None
    ) -> None:
        if (self.filter is None) or (selected is None):
            return
        out = add_fold_idx(
            self.filter / f"{selected.method}_selection_report.md", fold_idx=fold_idx
        )
        try:
            out.write_text(selected.to_markdown(), encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save filter selection report "
                f"to {out}. Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_pred_report(
        self, report: Optional[str], fold_idx: Optional[int] = None
    ) -> None:
        if (self.predictions is None) or (report is None):
            return
        out = add_fold_idx(self.predictions / "predictions_report.md", fold_idx=fold_idx)
        try:
            out.write_text(report, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save predictions report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_assoc_report(
        self, report: Optional[str], fold_idx: Optional[int] = None
    ) -> None:
        if (self.associations is None) or (report is None):
            return
        out = add_fold_idx(
            self.associations / "associations_report.md", fold_idx=fold_idx
        )
        try:
            out.write_text(report, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save associations report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_univariate_preds(
        self, preds: PredResults, fold_idx: Optional[int] = None
    ) -> None:
        if self.predictions is None:
            return
        try:
            preds.save_raw(self.predictions, fold_idx=fold_idx)
        except Exception as e:
            warn(
                "Got exception when attempting to save raw predictions. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )
        try:
            preds.save_tables(self.predictions, fold_idx=fold_idx)
        except Exception as e:
            warn(
                "Got exception when attempting to save prediction csv tables. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_univariate_assocs(
        self, assocs: AssocResults, fold_idx: Optional[int] = None
    ) -> None:
        if self.associations is None:
            return
        try:
            assocs.save_raw(self.associations, fold_idx=fold_idx)
        except Exception as e:
            warn(
                "Got exception when attempting to save raw associations. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )
        try:
            assocs.save_tables(self.associations, fold_idx=fold_idx)
        except Exception as e:
            warn(
                "Got exception when attempting to save association csv tables. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_feature_descriptions(
        self,
        desc_cont: Optional[DataFrame],
        desc_cat: Optional[DataFrame],
        desc_target: DataFrame,
    ) -> None:
        if self.descriptions is None:
            return

        conts = self.descriptions / "continuous_features.csv"
        cats = self.descriptions / "categorical_features.csv"
        target = self.descriptions / "target.csv"

        if desc_cont is not None:
            try:
                desc_cont.to_csv(conts)
            except Exception as e:
                warn(
                    "Got exception when attempting to save continuous feature "
                    f"descriptions. Details:\n{e}\n{traceback.format_exc()}"
                )

        if desc_cat is not None:
            try:
                desc_cat.to_csv(cats)
            except Exception as e:
                warn(
                    "Got exception when attempting to save categorical feature "
                    f"descriptions. Details:\n{e}\n{traceback.format_exc()}"
                )

        try:
            desc_target.to_csv(target)
        except Exception as e:
            warn(
                "Got exception when attempting to save target "
                f"descriptions. Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_prepared_raw(self, prepared: PreparedData) -> None:
        if self.prepared is None:
            return
        try:
            prepared.save_raw(self.prepared)
        except Exception as e:
            warn(
                "Got exception when attempting to prepared data. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_prep_report(self, report: Optional[str]) -> None:
        if (self.prepared is None) or (report is None):
            return

        out = self.prepared / "preparation_report.md"
        try:
            out.write_text(report, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save preparation report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_inspect_reports(self, inspection: InspectionResults) -> None:
        if self.inspection is None:
            return
        try:
            short = inspection.short_report()
            full = inspection.full_report()
            out_short = self.inspection / "short_inspection_report.md"
            out_full = self.inspection / "full_inspection_report.md"
            out_short.write_text(short, encoding="utf-8")
            if full is not None:
                out_full.write_text(full, encoding="utf-8")
        except Exception as e:
            warn(
                "Got exception when attempting to save inspection reports. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_inspect_tables(self, inspection: InspectionResults) -> None:
        if self.inspection is None:
            return
        try:
            df = inspection.basic_df()
            out = self.inspection / "inferred_types.csv"
            df.to_csv(out)
        except Exception as e:
            warn(
                "Got exception when attempting to save inspection table(s). "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def make_summary_report(self) -> Path:
        if self.results is None:
            raise ValueError("Missing results directory.")

        tables = sorted(self.results.rglob("*final_performances*.csv"))
        if len(tables) == 1:
            return tables[0]

        dfs = [pd.read_csv(table, index_col=0) for table in tables]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        sorter = "acc" if "acc" in df["metric"].values else "mae"
        ascending = sorter != "acc"
        pivot_index = ["model", "selection", "test_idx"]

        cols = ["holdout", "5-fold", "trainset"]
        final_tables = {}
        for valset in cols:
            g = (
                df.drop(columns=["embed_selector"])
                .pivot(columns="metric", values=valset, index=pivot_index)
                .reset_index()
                .drop(columns="test_idx")
                .groupby(["model", "selection"])
            )
            means = g.mean()
            mins = g.min()
            maxs = g.max()
            rngs = mins.round(3).map(lambda x: f"{x:0.3f}–") + maxs.round(3).map(
                lambda x: f"{x:0.3f}"
            )
            rngs.columns = rngs.columns.map(lambda col: f"{col}_rng")
            sort_cols = []
            for i in range(len(means.columns)):
                sort_cols.append(means.columns[i])
                sort_cols.append(rngs.columns[i])
            full = pd.concat([means.round(3), rngs], axis=1).loc[:, sort_cols]

            clean = (
                full.reset_index()
                .sort_values(by=sorter, ascending=ascending)
                .reset_index(drop=True)
            )
            final_tables[valset] = clean

        tab_train = final_tables["trainset"].to_markdown(
            tablefmt="simple", floatfmt="0.3f", index=False
        )

        tab_hold = final_tables["holdout"].to_markdown(
            tablefmt="simple", floatfmt="0.3f", index=False
        )
        tab_fold = final_tables["5-fold"].to_markdown(
            tablefmt="simple", floatfmt="0.3f", index=False
        )
        text = (
            "# Final Model Performances Summary Across Test Sets\n\n"
            "## Holdout set performances\n\n"
            f"{tab_hold}\n\n"
            "## 5-fold performance on holdout sets\n\n"
            f"{tab_fold}\n\n"
            "## Training set performances\n\n"
            f"{tab_train}\n\n"
        )
        outfile = self.results / "results_report.md"
        outfile.write_text(text)
        return outfile

    def cleanup(self) -> None:
        if self.root is None:
            return
        try:
            rmtree(self.root)
        except Exception:
            warn(
                f"Error during cleanup. Some files may remain at {self.root}. "
                f"Details:\n{traceback.format_exc()}"
            )


def get_hash(args: dict[str, Any], ignores: Optional[list[str]] = None) -> str:
    ignores = [] if ignores is None else ignores
    to_hash = {**args}
    for ignore in ignores:
        if ignore in to_hash:
            to_hash.pop(ignore)

    # quick and dirty hashing for caching  https://stackoverflow.com/a/1151705
    # we are not really worried about collisions with the tiny amount of
    # comparisons / different combinations we have here
    hsh = md5(str(tuple(sorted(to_hash.items()))).encode()).hexdigest()
    return hsh


def add_fold_idx(path: Path, fold_idx: Optional[int]) -> Path:
    if fold_idx is None:
        return path
    fname = path.name
    exts = "".join(path.suffixes)
    fname = fname.replace(exts, "")
    fname = f"{fname}_{fold_idx:02d}{exts}"
    newparent = path.parent / f"test{fold_idx:02d}"
    newparent.mkdir(exist_ok=True, parents=True)
    out = newparent / fname
    return out
