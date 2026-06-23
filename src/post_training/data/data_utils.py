import ast
import math
import operator
import re

RE_NUMBER = re.compile(  # A
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def get_last_boxed(text):
    """Return the content inside the last ``\\boxed{...}`` (or ``\\fbox{...}``).

    Uses balanced-brace matching so nested braces such as
    ``\\boxed{\\frac{1}{2}}`` are handled correctly. Returns ``None`` when no
    (balanced) boxed expression is present.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    brace_start = text.find("{", idx)
    if brace_start < 0:
        # space form, e.g. "\boxed 42" -> take the next token
        tail = text[idx:].split(None, 1)
        return tail[1].split("$", 1)[0].strip() if len(tail) > 1 else None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i]
    return None  # unbalanced braces


def extract_final_candidate(text, fallback="number_then_full"):
    result = ""  # B
    if text:  # C
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        # D
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]  # E
            elif fallback == "number_then_full":
                result = text  # F
    return result


LATEX_FIXES = [  # A
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"·|×", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # B
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",  # C
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",  # C
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",  # C
}


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
    # D
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)  # D
    text = re.sub(r"\^\s*\\circ", "", text)  # E
    text = text.replace("°", "")  # E
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)  # F
    if match:
        text = match.group("x")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)  # G
    for pat, rep in LATEX_FIXES:  # H
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)
    # I
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    # J
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    # K
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )
    # L
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
    return text.replace("{", "").replace("}", "").strip().lower()


def split_into_parts(text):
    result = [text]
    if text:  # A
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            items = [p.strip() for p in text[1:-1].split(",")]  # B
            if all(items):
                result = items
            else:  # C
                result = []
    return result


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
    ):
        return math.sqrt(_eval_node(node.args[0]))
    raise ValueError("unsupported expression")


def _safe_eval_number(s):
    """Evaluate a normalized arithmetic expression to a float, or return None.

    Only arithmetic and ``sqrt`` are allowed (no arbitrary code execution);
    anything symbolic (variables, unknown functions) yields ``None``.
    """
    if not s:
        return None
    try:
        return float(_eval_node(ast.parse(s, mode="eval").body))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return None


def equality_check(gt, pred):
    """True if two normalized answer parts match by string or numeric value."""
    if gt == pred:
        return True
    a, b = _safe_eval_number(gt), _safe_eval_number(pred)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return False


def grade_answer(pred_text, gt_text):
    result = False  # A
    if pred_text is not None and gt_text is not None:  # B
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )
        if (gt_parts and pred_parts  # C
                and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  # D
    return result
