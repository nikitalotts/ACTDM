#!/bin/bash
# Отчет о занятом месте: итог, самые тяжелые папки и самые большие файлы.
#
# Использование:
#   ./disk_report.sh                  по домашней директории, глубина 3
#   ./disk_report.sh /path            по указанному пути
#   ./disk_report.sh /path 4          с другой глубиной обхода папок
#
# Скрытые папки (.cache, .conda, ...) тоже учитываются.

TARGET="${1:-$HOME}"
DEPTH="${2:-3}"

echo "=== Всего занято: ${TARGET} ==="
du -sh "${TARGET}" 2>/dev/null

echo
echo "=== Папки до глубины ${DEPTH}, топ-40 по размеру ==="
du -h --max-depth="${DEPTH}" "${TARGET}" 2>/dev/null | sort -rh | head -40

echo
echo "=== Файлы крупнее 100M, топ-40 ==="
find "${TARGET}" -xdev -type f -size +100M -exec du -h {} + 2>/dev/null | sort -rh | head -40

echo
echo "=== Квота (если на кластере она есть) ==="
quota -s 2>/dev/null || df -h "${TARGET}" 2>/dev/null
