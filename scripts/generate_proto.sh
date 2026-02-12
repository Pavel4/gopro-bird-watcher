#!/bin/bash
# Генерация Python-кода из proto-файлов
# Запуск: bash scripts/generate_proto.sh

set -e

cd "$(dirname "$0")/.."

PROTO_DIR="proto"
OUT_DIR="detector/generated"

echo "🔧 Генерация Python-кода из proto..."

# Создаём выходную директорию
mkdir -p "$OUT_DIR"

# Генерация через grpcio-tools
python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/inference.proto"

# Создаём __init__.py для пакета
touch "$OUT_DIR/__init__.py"

# Фиксим импорт в сгенерированном gRPC-файле:
# grpc_tools генерирует "import inference_pb2",
# но нам нужен относительный импорт
GRPC_FILE="$OUT_DIR/inference_pb2_grpc.py"
if [ -f "$GRPC_FILE" ]; then
    sed -i 's/^import inference_pb2/from . import inference_pb2/' \
        "$GRPC_FILE" 2>/dev/null || \
    sed -i '' 's/^import inference_pb2/from . import inference_pb2/' \
        "$GRPC_FILE"
fi

echo "✅ Сгенерировано в $OUT_DIR/"
ls -la "$OUT_DIR/"
