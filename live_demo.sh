#!/bin/bash

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON="$PROJECT/.venv/bin/python"
OUTPUTS="$PROJECT/demo_terminal_outputs.txt"

# FULL_SHEET_XLSX="$PROJECT/data/templates/student/full_sheet.xlsx"
FULL_SHEET_CSV="$PROJECT/data/templates/student/full_sheet.csv"
# MIN_SHEET_XLSX="$PROJECT/data/templates/student/minimal_sheet.xlsx"
# MIN_SHEET_CSV="$PROJECT/data/templates/student/minimal_sheet.csv"

# $PYTHON df-analyze.py --spreadsheet "$MIN_SHEET_CSV" || (echo "Failed on min .csv file" && exit 1);
# $PYTHON df-analyze.py --spreadsheet "$MIN_SHEET_XLSX" || (echo "Failed on min .csv file" && exit 1);
$PYTHON df-analyze.py --spreadsheet "$FULL_SHEET_CSV" 2>&1 | tee "$OUTPUTS";
# $PYTHON df-analyze.py --spreadsheet "$FULL_SHEET_XLSX" || (echo "Failed on full .xlsx file" && exit 1);
