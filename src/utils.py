from pprint import pformat


class Debug:
    """Printing mixin a la https://doc.rust-lang.org/std/fmt/trait.Debug.html"""

    def __repr__(self) -> str:
        return "".join(pformat(self.__dict__, indent=2, width=80, compact=False))
