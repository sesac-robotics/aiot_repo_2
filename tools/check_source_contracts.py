"""통합 브랜치의 파싱·설명·의존 방향·import 경로 계약을 검사합니다.

스타일 도구를 대신하지 않습니다. 카메라·모델·DB를 import하지 않고 AST만 읽어
환경에 의존하지 않는 경계 회귀를 확인합니다. 전체 레거시 타입 완성률을 뜻하지 않습니다.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("Device", "GUI", "Server", "shared", "tools", "tests")
PURE_ROOTS = (
    "GUI/FitHelper/domain",
    "Server/AIFitService/domain",
    "Server/AIService/app/session",
)


def check_sources(root: Path = ROOT) -> tuple[int, list[str]]:
    """외부 코드를 실행하지 않고 소스를 검사해 파일 수와 위반 목록을 반환합니다."""
    count = 0
    failures = []
    for component in SOURCE_ROOTS:
        for path in sorted((root / component).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            count += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"{relative}:{exc.lineno}: syntax error")
                continue
            if not re.search("[가-힣]", ast.get_docstring(tree) or ""):
                failures.append(
                    f"{relative}: Korean module responsibility missing"
                )
            pure = any(
                relative.startswith(prefix + "/") for prefix in PURE_ROOTS
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(alias.name == "*" for alias in node.names):
                        failures.append(
                            f"{relative}:{node.lineno}: wildcard import"
                        )
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    modules = []
                for module in modules:
                    if pure and not (
                        module.split(".")[0] in sys.stdlib_module_names
                        or module.startswith(
                            (
                                "app.session",
                                "GUI.FitHelper.domain",
                                "Server.AIFitService.domain",
                            )
                        )
                    ):
                        failures.append(
                            f"{relative}:{node.lineno}: non-domain dependency {module}"
                        )
                if isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute
                ):
                    expression = ast.unparse(node.func)
                    if expression.startswith("sys.path."):
                        failures.append(
                            f"{relative}:{node.lineno}: sys.path mutation"
                        )
    return count, failures


def main() -> int:
    """실패 목록과 검사 범위를 출력하고 CI가 사용할 종료 코드를 반환합니다."""
    count, failures = check_sources()
    for failure in failures:
        print(failure)
    print(f"Source contracts: {count} Python files, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
