#!/usr/bin/env python3
"""
Вариант 9 (Конфигурационное управление) - реализация на Python.

Функционал:
- Этап 1: чтение XML-конфига, вывод параметров, обработка ошибок.
- Этап 2: получение прямых зависимостей для формата Rust (Cargo).
- Этап 3: построение графа зависимостей (BFS с рекурсией), фильтрация, циклы, тестовый режим.
- Этап 4: вывод обратных зависимостей для заданного пакета.
- Этап 5: генерация описания графа в формате D2 и попытка сохранения PNG через внешнюю утилиту `d2`.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from typing import Dict, Set, List, Optional
import subprocess

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    tomllib = None
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None


@dataclass
class Config:
    package_name: str
    repository_url: str
    use_test_repository: bool
    output_image: str
    filter_substring: Optional[str] = None


# ==========================
# XML-конфиг: чтение и валидация
# ==========================

def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    def text(tag: str, default: Optional[str] = None) -> Optional[str]:
        elem = root.find(tag)
        return elem.text.strip() if elem is not None and elem.text is not None else default

    package_name = text("packageName")
    repository_url = text("repositoryUrl")
    use_test_repo_str = text("useTestRepository", "false")
    output_image = text("outputImage", "graph.png")
    filter_substring = text("filterSubstring", "")

    use_test_repository = use_test_repo_str.lower() in ("1", "true", "yes", "y")

    return Config(
        package_name=package_name or "",
        repository_url=repository_url or "",
        use_test_repository=use_test_repository,
        output_image=output_image or "graph.png",
        filter_substring=filter_substring or None
    )


def validate_config(cfg: Config) -> List[str]:
    errors = []

    if not cfg.package_name:
        errors.append("package_name is empty")

    if not cfg.repository_url:
        errors.append("repository_url is empty")

    if not cfg.output_image:
        errors.append("output_image is empty")

    # Для фильтра строки нет жестких требований – можно пустую

    return errors


def print_config(cfg: Config) -> None:
    print("=== Конфигурация (ключ-значение) ===")
    for f in fields(cfg):
        print(f"{f.name} = {getattr(cfg, f.name)!r}")
    print("====================================")


# ==========================
# Загрузка данных: Cargo / тестовый граф
# ==========================

def load_cargo_dependencies(cargo_toml_path_or_url: str) -> List[str]:
    """
    Этап 2: извлечение прямых зависимостей из Cargo.toml.
    Для простоты считаем, что repository_url указывает либо:
      - на локальный файл Cargo.toml
      - на raw-URL Cargo.toml (http/https)
    """
    if tomllib is None:
        raise RuntimeError(
            "Модуль tomllib/tomli не найден. Установите tomli (pip install tomli) "
            "или используйте Python 3.11+."
        )

    if cargo_toml_path_or_url.startswith("http://") or cargo_toml_path_or_url.startswith("https://"):
        # Простейшая реализация через requests
        try:
            import requests
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Для загрузки Cargo.toml по URL требуется библиотека 'requests' (pip install requests)."
            ) from e

        resp = requests.get(cargo_toml_path_or_url, timeout=10)
        resp.raise_for_status()
        data = tomllib.loads(resp.content)
    else:
        # Локальный путь
        if not os.path.exists(cargo_toml_path_or_url):
            raise FileNotFoundError(f"Cargo.toml not found: {cargo_toml_path_or_url}")
        with open(cargo_toml_path_or_url, "rb") as f:
            data = tomllib.load(f)

    deps = data.get("dependencies", {})
    # dependencies может быть:
    #   "serde" = "1.0"
    #   serde = { version = "1.0", features = ["derive"] }
    # нас интересуют только имена
    return list(deps.keys())


def load_test_graph(path: str) -> Dict[str, Set[str]]:
    """
    Формат файла:
        A: B C
        B: C D
        C: D
        D:
    Строки, начинающиеся с #, игнорируются.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test graph file not found: {path}")

    graph: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid line in test graph: {line}")
            node_part, deps_part = line.split(":", 1)
            node = node_part.strip()
            deps = deps_part.strip().split() if deps_part.strip() else []
            graph.setdefault(node, set())
            for d in deps:
                graph[node].add(d)
                graph.setdefault(d, set())  # чтобы узлы без детей тоже были в графе
    return graph


# ==========================
# Граф: BFS с рекурсией, фильтрация и циклы
# ==========================

def bfs_recursive(
    queue: List[str],
    graph: Dict[str, Set[str]],
    visited: Set[str],
    filter_substring: Optional[str],
    order: List[str]
) -> None:
    """Реализация BFS с рекурсией (по требованию варианта)."""
    if not queue:
        return

    node = queue.pop(0)

    if node in visited:
        return

    if filter_substring and filter_substring in node:
        # узел игнорируем полностью
        visited.add(node)
        bfs_recursive(queue, graph, visited, filter_substring, order)
        return

    visited.add(node)
    order.append(node)

    for neigh in sorted(graph.get(node, [])):
        if neigh not in visited:
            queue.append(neigh)

    bfs_recursive(queue, graph, visited, filter_substring, order)


def build_dependency_graph_from_test(
    cfg: Config,
    start: str
) -> Dict[str, Set[str]]:
    """Построение графа зависимостей из тестового файла + BFS-порядок (для демонстрации)."""
    graph = load_test_graph(cfg.repository_url)
    visited: Set[str] = set()
    order: List[str] = []
    bfs_recursive([start], graph, visited, cfg.filter_substring, order)

    print("BFS-порядок обхода (с учетом фильтра и циклов):")
    print(" -> ".join(order) if order else "(пусто)")
    return graph


# ==========================
# Обратные зависимости (этап 4)
# ==========================

def build_reverse_graph(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    rev: Dict[str, Set[str]] = {}
    for src, targets in graph.items():
        for t in targets:
            rev.setdefault(t, set()).add(src)
        rev.setdefault(src, rev.get(src, set()))
    return rev


def find_reverse_dependencies(
    graph: Dict[str, Set[str]],
    target: str,
    filter_substring: Optional[str]
) -> List[str]:
    """Поиск всех пакетов, которые зависят от target (обратные зависимости)."""
    rev = build_reverse_graph(graph)
    visited: Set[str] = set()
    order: List[str] = []
    bfs_recursive([target], rev, visited, filter_substring, order)
    # Первый элемент - сам target, его можно убрать
    return [node for node in order if node != target]


# ==========================
# Визуализация: D2 + PNG
# ==========================

def graph_to_d2(graph: Dict[str, Set[str]], start: Optional[str] = None) -> str:
    """
    Генерация текста диаграммы в формате D2.
    Пример:
        A -> B
        A -> C
    """
    lines = []
    lines.append("graph {")
    for src, targets in sorted(graph.items()):
        for dst in sorted(targets):
            lines.append(f"  {src} -> {dst}")
        if not targets:
            # чтобы孤чный узел тоже попал в диаграмму
            lines.append(f"  {src}")
    lines.append("}")
    return "\n".join(lines)


def save_d2_and_png(d2_text: str, d2_path: str, png_path: str) -> None:
    with open(d2_path, "w", encoding="utf-8") as f:
        f.write(d2_text)
    print(f"D2-описание графа сохранено в {d2_path}")

    # Пытаемся вызвать внешнюю утилиту d2 (если установлена)
    try:
        result = subprocess.run(
            ["d2", d2_path, png_path],
            check=False,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"PNG-изображение графа сохранено в {png_path}")
        else:
            print("Не удалось сгенерировать PNG через утилиту d2.")
            print("Установите d2 (https://d2lang.com) и запустите:")
            print(f"  d2 {d2_path} {png_path}")
            if result.stderr:
                print("stderr:", result.stderr)
    except FileNotFoundError:
        print("Утилита 'd2' не найдена в PATH.")
        print("Установите d2 и выполните вручную:")
        print(f"  d2 {d2_path} {png_path}")


# ==========================
# Основные этапы
# ==========================

def stage1(cfg: Config) -> None:
    """Минимальный прототип с конфигурацией."""
    print_config(cfg)
    errors = validate_config(cfg)
    if errors:
        print("\nОбнаружены ошибки конфигурации:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    else:
        print("\nКонфигурация валидна.")


def stage2(cfg: Config) -> None:
    """Сбор прямых зависимостей (Cargo)."""
    if cfg.use_test_repository:
        print("ВНИМАНИЕ: use_test_repository=true, но Этап 2 предполагает реальный Cargo.toml.")
        print("Будет использован путь из repository_url как путь к Cargo.toml.")
    deps = load_cargo_dependencies(cfg.repository_url)
    print("Прямые зависимости (из Cargo.toml):")
    for d in deps:
        print(f" - {d}")


def stage3(cfg: Config) -> Dict[str, Set[str]]:
    """Построение графа зависимостей (BFS с рекурсией, режим тестирования)."""
    if not cfg.use_test_repository:
        print("Этап 3: для демонстрации удобнее использовать тестовый репозиторий (useTestRepository=true).")
        print("Сейчас будет выполнена работа с тестовым файлом графа.")
    graph = build_dependency_graph_from_test(cfg, cfg.package_name)
    return graph


def stage4(cfg: Config, graph: Dict[str, Set[str]]) -> None:
    """Вывод обратных зависимостей."""
    rev_deps = find_reverse_dependencies(graph, cfg.package_name, cfg.filter_substring)
    print(f"Обратные зависимости для пакета {cfg.package_name!r}:")
    if rev_deps:
        for d in rev_deps:
            print(f" - {d}")
    else:
        print(" (нет обратных зависимостей)")


def stage5(cfg: Config, graph: Dict[str, Set[str]]) -> None:
    """Визуализация графа зависимостей в формате D2 + PNG."""
    d2_text = graph_to_d2(graph, start=cfg.package_name)
    d2_path = os.path.splitext(cfg.output_image)[0] + ".d2"
    png_path = cfg.output_image
    save_d2_and_png(d2_text, d2_path, png_path)


# ==========================
# CLI
# ==========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Инструмент визуализации графа зависимостей (Вариант 9, Rust/Cargo)"
    )
    p.add_argument(
        "-c", "--config",
        default="config.xml",
        help="Путь к XML-конфигу (по умолчанию config.xml)"
    )
    p.add_argument(
        "-s", "--stage",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Какой этап выполнить (1-5 или all)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Ошибка при чтении конфигурации: {e}")
        sys.exit(1)

    # Этап 1 выполняем всегда перед остальными, чтобы убедиться, что конфиг ок
    print("\n=== ЭТАП 1 ===")
    stage1(cfg)

    graph: Dict[str, Set[str]] = {}

    if args.stage in ("2", "all"):
        print("\n=== ЭТАП 2 ===")
        try:
            stage2(cfg)
        except Exception as e:
            print(f"Ошибка на этапе 2: {e}")

    if args.stage in ("3", "all"):
        print("\n=== ЭТАП 3 ===")
        try:
            graph = stage3(cfg)
        except Exception as e:
            print(f"Ошибка на этапе 3: {e}")
            graph = {}

    if (args.stage in ("4", "5", "all")) and not graph:
        # если граф ещё не построен, а он нужен для этапов 4/5 — пробуем построить
        try:
            graph = stage3(cfg)
        except Exception as e:
            print(f"Не удалось построить граф для этапов 4/5: {e}")
            graph = {}

    if args.stage in ("4", "all") and graph:
        print("\n=== ЭТАП 4 ===")
        try:
            stage4(cfg, graph)
        except Exception as e:
            print(f"Ошибка на этапе 4: {e}")

    if args.stage in ("5", "all") and graph:
        print("\n=== ЭТАП 5 ===")
        try:
            stage5(cfg, graph)
        except Exception as e:
            print(f"Ошибка на этапе 5: {e}")


if __name__ == "__main__":
    main()
