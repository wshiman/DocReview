__all__ = ["apply_annotations", "parse_document", "review_paragraphs"]


def __getattr__(name):
    if name == "apply_annotations":
        from .annotators import apply_annotations

        return apply_annotations
    if name == "parse_document":
        from .parsers import parse_document

        return parse_document
    if name == "review_paragraphs":
        from .reviewer import review_paragraphs

        return review_paragraphs
    raise AttributeError(name)
