#!/usr/bin/env python3
"""
Скачивание фотографий птиц с iNaturalist для обучения
классификатора видов.

Использует iNaturalist API v1 напрямую (без pyinaturalist)
для загрузки research-grade наблюдений 10 видов
московских кормушечных птиц.

Использование:
    pip install -r scripts/requirements-train.txt
    python scripts/download_bird_images.py \
        --output_dir data/train \
        --per_species 200

Или с ограничением по географии:
    python scripts/download_bird_images.py \
        --output_dir data/train \
        --per_species 200 \
        --geo europe
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Топ-10 видов московских кормушечных птиц ===
# taxon_id: iNaturalist taxon ID
# species_en: английское название (для директорий)
# species_ru: русское название
# class_id: индекс класса в species_labels.json
SPECIES = [
    {
        "class_id": 0,
        "taxon_id": 203153,
        "species_en": "great_tit",
        "species_ru": "Большая синица",
        "latin": "Parus major",
    },
    {
        "class_id": 1,
        "taxon_id": 144849,
        "species_en": "blue_tit",
        "species_ru": "Лазоревка",
        "latin": "Cyanistes caeruleus",
    },
    {
        "class_id": 2,
        "taxon_id": 13858,
        "species_en": "house_sparrow",
        "species_ru": "Домовый воробей",
        "latin": "Passer domesticus",
    },
    {
        "class_id": 3,
        "taxon_id": 13851,
        "species_en": "tree_sparrow",
        "species_ru": "Полевой воробей",
        "latin": "Passer montanus",
    },
    {
        "class_id": 4,
        "taxon_id": 14824,
        "species_en": "nuthatch",
        "species_ru": "Поползень",
        "latin": "Sitta europaea",
    },
    {
        "class_id": 5,
        "taxon_id": 9462,
        "species_en": "bullfinch",
        "species_ru": "Снегирь",
        "latin": "Pyrrhula pyrrhula",
    },
    {
        "class_id": 6,
        "taxon_id": 17871,
        "species_en": "great_spotted_woodpecker",
        "species_ru": "Большой пёстрый дятел",
        "latin": "Dendrocopos major",
    },
    {
        "class_id": 7,
        "taxon_id": 145360,
        "species_en": "greenfinch",
        "species_ru": "Зеленушка",
        "latin": "Chloris chloris",
    },
    {
        "class_id": 8,
        "taxon_id": 145303,
        "species_en": "siskin",
        "species_ru": "Чиж",
        "latin": "Spinus spinus",
    },
    {
        "class_id": 9,
        "taxon_id": 7429,
        "species_en": "waxwing",
        "species_ru": "Свиристель",
        "latin": "Bombycilla garrulus",
    },
]

# Географические фильтры
GEO_FILTERS = {
    "moscow": {
        "lat": 55.7558,
        "lng": 37.6173,
        "radius": 100,  # km
    },
    "europe": {
        # Bounding box: Европа (широкий охват)
        "nelat": 70.0,
        "nelng": 50.0,
        "swlat": 35.0,
        "swlng": -10.0,
    },
    "russia": {
        "nelat": 70.0,
        "nelng": 60.0,
        "swlat": 45.0,
        "swlng": 25.0,
    },
    "none": {},  # Без географического фильтра
}

# iNaturalist API
INAT_API_BASE = "https://api.inaturalist.org/v1"
# Рекомендуемый rate limit: 60 req/min
RATE_LIMIT_DELAY = 1.1  # секунд между запросами API
DOWNLOAD_THREADS = 4
DOWNLOAD_TIMEOUT = 30  # секунд на скачивание фото
USER_AGENT = (
    "GoPro-Bird-Watcher/1.0 "
    "(species classifier training data)"
)


def fetch_observations(
    taxon_id: int,
    per_page: int = 50,
    page: int = 1,
    geo_filter: dict = None,
) -> dict:
    """
    Получить наблюдения из iNaturalist API.

    Args:
        taxon_id: ID таксона
        per_page: количество на страницу (макс 200)
        page: номер страницы
        geo_filter: географический фильтр

    Returns:
        JSON ответ API
    """
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(per_page, 200),
        "page": page,
        "order": "desc",
        "order_by": "votes",
        "locale": "en",
    }

    if geo_filter:
        params.update(geo_filter)

    url = (
        f"{INAT_API_BASE}/observations?"
        + urllib.parse.urlencode(params)
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urllib.request.urlopen(
            req, timeout=30
        ) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        logger.error(
            f"HTTP error {e.code}: {e.reason}"
        )
        if e.code == 429:
            logger.warning(
                "Rate limited! Waiting 60s..."
            )
            time.sleep(60)
            return fetch_observations(
                taxon_id, per_page, page,
                geo_filter,
            )
        return {"results": [], "total_results": 0}
    except Exception as e:
        logger.error(
            f"API request failed: {e}"
        )
        return {"results": [], "total_results": 0}


def extract_photo_urls(observations: list) -> list:
    """
    Извлечь URL фотографий из наблюдений.

    Берёт только первое фото каждого наблюдения
    (обычно лучшее). Заменяет размер на "medium"
    (500px) для баланса качества/размера.

    Returns:
        Список словарей {url, observation_id}
    """
    photos = []
    for obs in observations:
        obs_id = obs.get("id", 0)
        obs_photos = obs.get("photos", [])

        if not obs_photos:
            continue

        # Берём первое фото (обычно лучшее)
        photo = obs_photos[0]
        url = photo.get("url", "")

        if not url:
            continue

        # Заменяем размер на medium (500px)
        # iNat URL: .../square.jpg -> .../medium.jpg
        url = url.replace("/square.", "/medium.")

        photos.append({
            "url": url,
            "observation_id": obs_id,
            "photo_id": photo.get("id", 0),
        })

    return photos


def download_photo(
    url: str,
    filepath: str,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> bool:
    """
    Скачать фото по URL в файл.

    Returns:
        True если успешно
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(
            req, timeout=timeout
        ) as resp:
            data = resp.read()

        # Проверяем что это изображение (>1KB)
        if len(data) < 1024:
            return False

        with open(filepath, "wb") as f:
            f.write(data)

        return True
    except Exception as e:
        logger.debug(
            f"Failed to download {url}: {e}"
        )
        return False


def collect_species_photos(
    species: dict,
    output_dir: str,
    target_count: int = 200,
    geo_filter: dict = None,
) -> list:
    """
    Собрать фотографии одного вида.

    Args:
        species: словарь с данными вида
        output_dir: корневая директория данных
        target_count: целевое количество фото
        geo_filter: географический фильтр

    Returns:
        Список словарей для manifest.csv
    """
    class_id = species["class_id"]
    species_en = species["species_en"]
    species_ru = species["species_ru"]
    taxon_id = species["taxon_id"]

    # Директория для вида
    species_dir = os.path.join(
        output_dir,
        f"{class_id}_{species_en}",
    )
    os.makedirs(species_dir, exist_ok=True)

    # Проверяем сколько уже скачано
    existing = [
        f for f in os.listdir(species_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(existing) >= target_count:
        logger.info(
            f"  {species_en}: уже есть "
            f"{len(existing)} фото, пропускаем"
        )
        return []

    logger.info(
        f"  Загрузка {species_ru} "
        f"({species_en}, taxon={taxon_id})..."
    )

    # Собираем URL фотографий
    all_photo_urls = []
    page = 1
    max_pages = 20  # Ограничение по страницам

    while (
        len(all_photo_urls) < target_count
        and page <= max_pages
    ):
        resp = fetch_observations(
            taxon_id=taxon_id,
            per_page=200,
            page=page,
            geo_filter=geo_filter,
        )

        results = resp.get("results", [])
        if not results:
            break

        photos = extract_photo_urls(results)
        all_photo_urls.extend(photos)

        total = resp.get("total_results", 0)
        logger.info(
            f"    Страница {page}: "
            f"+{len(photos)} фото "
            f"(всего {len(all_photo_urls)}/"
            f"{target_count}, "
            f"доступно {total})"
        )

        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    # Обрезаем до целевого количества
    photo_urls = all_photo_urls[:target_count]

    if not photo_urls:
        logger.warning(
            f"  {species_en}: "
            f"не найдено фотографий!"
        )
        return []

    # Скачиваем параллельно
    logger.info(
        f"  Скачивание {len(photo_urls)} фото "
        f"({DOWNLOAD_THREADS} потоков)..."
    )

    manifest_rows = []
    downloaded = len(existing)
    failed = 0

    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_THREADS
    ) as pool:
        futures = {}
        for idx, photo in enumerate(photo_urls):
            filename = f"{idx + 1:04d}.jpg"
            filepath = os.path.join(
                species_dir, filename
            )

            # Пропускаем если файл уже есть
            if os.path.exists(filepath):
                manifest_rows.append({
                    "path": os.path.relpath(
                        filepath, output_dir
                    ),
                    "class_id": class_id,
                    "species_en": species_en,
                    "species_ru": species_ru,
                    "observation_id": (
                        photo["observation_id"]
                    ),
                })
                continue

            future = pool.submit(
                download_photo,
                photo["url"],
                filepath,
            )
            futures[future] = {
                "filepath": filepath,
                "photo": photo,
                "filename": filename,
            }

        for future in as_completed(futures):
            info = futures[future]
            try:
                success = future.result()
                if success:
                    downloaded += 1
                    manifest_rows.append({
                        "path": os.path.relpath(
                            info["filepath"],
                            output_dir,
                        ),
                        "class_id": class_id,
                        "species_en": species_en,
                        "species_ru": species_ru,
                        "observation_id": (
                            info["photo"][
                                "observation_id"
                            ]
                        ),
                    })
                else:
                    failed += 1
            except Exception:
                failed += 1

    logger.info(
        f"  {species_en}: "
        f"скачано {downloaded}, "
        f"ошибок {failed}"
    )

    return manifest_rows


def write_manifest(
    manifest_rows: list,
    output_dir: str,
):
    """Записать CSV-манифест всех скачанных фото."""
    manifest_path = os.path.join(
        output_dir, "manifest.csv"
    )
    fieldnames = [
        "path",
        "class_id",
        "species_en",
        "species_ru",
        "observation_id",
    ]

    with open(
        manifest_path, "w",
        newline="", encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        f"Манифест сохранён: {manifest_path} "
        f"({len(manifest_rows)} записей)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Скачивание фотографий птиц "
            "с iNaturalist для обучения "
            "классификатора видов"
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train",
        help="Директория для сохранения фото",
    )
    parser.add_argument(
        "--per_species",
        type=int,
        default=200,
        help="Количество фото на вид (по умолч. 200)",
    )
    parser.add_argument(
        "--geo",
        type=str,
        default="europe",
        choices=list(GEO_FILTERS.keys()),
        help="Географический фильтр",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help=(
            "Конкретный вид (english name) "
            "для загрузки, напр. 'great_tit'"
        ),
    )

    args = parser.parse_args()

    # Создаём выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)

    geo_filter = GEO_FILTERS.get(args.geo, {})
    geo_name = args.geo

    logger.info(
        f"=== Загрузка данных для обучения ==="
    )
    logger.info(
        f"Директория: {args.output_dir}"
    )
    logger.info(
        f"Фото на вид: {args.per_species}"
    )
    logger.info(
        f"Гео-фильтр: {geo_name}"
    )

    # Фильтруем виды если указан конкретный
    species_list = SPECIES
    if args.species:
        species_list = [
            s for s in SPECIES
            if s["species_en"] == args.species
        ]
        if not species_list:
            logger.error(
                f"Вид '{args.species}' не найден. "
                f"Доступные: "
                + ", ".join(
                    s["species_en"] for s in SPECIES
                )
            )
            sys.exit(1)

    all_manifest_rows = []

    for i, species in enumerate(species_list):
        logger.info(
            f"\n[{i + 1}/{len(species_list)}] "
            f"{species['species_ru']} "
            f"({species['latin']})"
        )

        rows = collect_species_photos(
            species=species,
            output_dir=args.output_dir,
            target_count=args.per_species,
            geo_filter=geo_filter,
        )
        all_manifest_rows.extend(rows)

        # Пауза между видами
        if i < len(species_list) - 1:
            time.sleep(2)

    # Сохраняем манифест
    write_manifest(
        all_manifest_rows, args.output_dir
    )

    # Статистика
    logger.info(f"\n=== Итого ===")
    species_counts = {}
    for row in all_manifest_rows:
        name = row["species_en"]
        species_counts[name] = (
            species_counts.get(name, 0) + 1
        )

    for name, count in sorted(
        species_counts.items()
    ):
        logger.info(f"  {name}: {count} фото")

    total = sum(species_counts.values())
    logger.info(f"  Всего: {total} фото")


if __name__ == "__main__":
    main()
