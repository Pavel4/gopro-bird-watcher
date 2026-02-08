"""
Тесты для scripts/download_bird_images.py

Тестируем: SPECIES list, extract_photo_urls,
geo_filters, write_manifest, download_photo.
"""

import csv
import os
from unittest.mock import (
    MagicMock, patch, mock_open,
)

import pytest

from download_bird_images import (
    SPECIES,
    GEO_FILTERS,
    extract_photo_urls,
    write_manifest,
    download_photo,
)


class TestSpeciesList:
    def test_species_count(self):
        """Ровно 10 видов."""
        assert len(SPECIES) == 10

    def test_unique_class_ids(self):
        """class_id уникальны."""
        ids = [s["class_id"] for s in SPECIES]
        assert len(ids) == len(set(ids))

    def test_unique_taxon_ids(self):
        """taxon_id уникальны."""
        ids = [s["taxon_id"] for s in SPECIES]
        assert len(ids) == len(set(ids))

    def test_sequential_class_ids(self):
        """class_id от 0 до 9."""
        ids = sorted(
            s["class_id"] for s in SPECIES
        )
        assert ids == list(range(10))

    def test_required_fields(self):
        """Каждый вид имеет все поля."""
        required = {
            "class_id", "taxon_id",
            "species_en", "species_ru", "latin",
        }
        for sp in SPECIES:
            missing = required - set(sp.keys())
            assert not missing, (
                f"{sp['species_en']}: "
                f"нет полей {missing}"
            )

    def test_known_species(self):
        """Проверяем конкретные виды."""
        names_en = [
            s["species_en"] for s in SPECIES
        ]
        assert "great_tit" in names_en
        assert "blue_tit" in names_en
        assert "house_sparrow" in names_en
        assert "nuthatch" in names_en
        assert "bullfinch" in names_en


class TestGeoFilters:
    def test_all_filters_exist(self):
        """Все гео-фильтры определены."""
        expected = [
            "moscow", "europe", "russia", "none",
        ]
        for name in expected:
            assert name in GEO_FILTERS

    def test_moscow_has_coordinates(self):
        """Москва имеет lat/lng/radius."""
        moscow = GEO_FILTERS["moscow"]
        assert "lat" in moscow
        assert "lng" in moscow
        assert "radius" in moscow
        assert moscow["lat"] == pytest.approx(
            55.7558, abs=0.01
        )

    def test_none_is_empty(self):
        """Фильтр 'none' пустой."""
        assert GEO_FILTERS["none"] == {}


class TestExtractPhotoUrls:
    def test_extract_basic(self):
        """Извлечение URL из наблюдений."""
        observations = [
            {
                "id": 100,
                "photos": [
                    {
                        "id": 1,
                        "url": (
                            "https://static.inaturalist"
                            ".org/photos/1/square.jpg"
                        ),
                    }
                ],
            },
            {
                "id": 200,
                "photos": [
                    {
                        "id": 2,
                        "url": (
                            "https://static.inaturalist"
                            ".org/photos/2/square.jpg"
                        ),
                    }
                ],
            },
        ]
        result = extract_photo_urls(observations)
        assert len(result) == 2
        assert result[0]["observation_id"] == 100
        # square -> medium замена
        assert "medium" in result[0]["url"]
        assert "square" not in result[0]["url"]

    def test_extract_empty(self):
        """Пустой список наблюдений."""
        result = extract_photo_urls([])
        assert result == []

    def test_extract_no_photos(self):
        """Наблюдение без фото пропускается."""
        observations = [
            {"id": 100, "photos": []},
            {"id": 200},
        ]
        result = extract_photo_urls(observations)
        assert result == []

    def test_extract_takes_first_photo(self):
        """Берёт только первое фото."""
        observations = [
            {
                "id": 100,
                "photos": [
                    {
                        "id": 1,
                        "url": (
                            "https://ex.com/1/"
                            "square.jpg"
                        ),
                    },
                    {
                        "id": 2,
                        "url": (
                            "https://ex.com/2/"
                            "square.jpg"
                        ),
                    },
                ],
            },
        ]
        result = extract_photo_urls(observations)
        assert len(result) == 1
        assert result[0]["photo_id"] == 1


class TestWriteManifest:
    def test_write_manifest(self, tmp_dir):
        """write_manifest создаёт корректный CSV."""
        rows = [
            {
                "path": "0_great_tit/0001.jpg",
                "class_id": 0,
                "species_en": "great_tit",
                "species_ru": "Большая синица",
                "observation_id": 12345,
            },
            {
                "path": "1_blue_tit/0001.jpg",
                "class_id": 1,
                "species_en": "blue_tit",
                "species_ru": "Лазоревка",
                "observation_id": 67890,
            },
        ]
        write_manifest(rows, tmp_dir)

        manifest_path = os.path.join(
            tmp_dir, "manifest.csv"
        )
        assert os.path.exists(manifest_path)

        with open(
            manifest_path, "r", encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            loaded = list(reader)

        assert len(loaded) == 2
        assert loaded[0]["species_en"] == "great_tit"
        assert loaded[1]["class_id"] == "1"


class TestDownloadPhoto:
    @patch("download_bird_images.urllib.request")
    def test_download_success(
        self, mock_urllib, tmp_dir,
    ):
        """Успешное скачивание фото."""
        # Мокаем ответ > 1KB
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"x" * 2048
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(
            return_value=False
        )
        mock_urllib.urlopen.return_value = mock_resp

        filepath = os.path.join(
            tmp_dir, "test.jpg"
        )
        result = download_photo(
            "https://example.com/photo.jpg",
            filepath,
        )
        assert result is True
        assert os.path.exists(filepath)

    @patch("download_bird_images.urllib.request")
    def test_download_too_small(
        self, mock_urllib, tmp_dir,
    ):
        """Файл < 1KB отклоняется."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"tiny"
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(
            return_value=False
        )
        mock_urllib.urlopen.return_value = mock_resp

        filepath = os.path.join(
            tmp_dir, "tiny.jpg"
        )
        result = download_photo(
            "https://example.com/tiny.jpg",
            filepath,
        )
        assert result is False
