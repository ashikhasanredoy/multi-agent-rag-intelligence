import ast
import math
from typing import Dict, Any
from app.utils.logging import logger


class CodeTool:
    """Safe evaluation and code analysis tool."""

    @staticmethod
    def evaluate_math_expression(expr: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions."""
        try:
            # Clean expression
            clean_expr = expr.strip()
            allowed_names = {
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }
            node = ast.parse(clean_expr, mode='eval')
            code = compile(node, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return {"success": True, "result": result, "expression": clean_expr}
        except Exception as e:
            return {"success": False, "error": str(e), "expression": expr}


code_tool = CodeTool()
