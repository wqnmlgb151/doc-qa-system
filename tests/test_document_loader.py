import pytest

from document_loader import (
    DocumentLoaderError,
    UnsupportedFileTypeError,
    EmptyDocumentError,
    DocumentParseError,
    load_file,
)


class TestCustomExceptions:
    def test_document_loader_error_has_message_and_detail(self):
        e = DocumentLoaderError("简短消息", "详细说明")
        assert str(e) == "简短消息"
        assert e.detail == "详细说明"

    def test_unsupported_file_type_error(self):
        e = UnsupportedFileTypeError(".xyz")
        assert "不支持的文件类型" in str(e)
        assert ".xyz" in str(e)
        assert ".pdf" in e.detail

    def test_empty_document_error(self):
        e = EmptyDocumentError("OCR 未能识别")
        assert "为空" in str(e)
        assert "OCR 未能识别" in e.detail

    def test_document_parse_error(self):
        e = DocumentParseError(".doc 格式无法解析")
        assert "无法解析" in str(e)
        assert ".doc" in e.detail

    def test_exceptions_are_importable_from_module(self):
        # Verify public API
        from document_loader import (
            DocumentLoaderError,
            UnsupportedFileTypeError,
            EmptyDocumentError,
            DocumentParseError,
        )
        assert issubclass(UnsupportedFileTypeError, DocumentLoaderError)
        assert issubclass(EmptyDocumentError, DocumentLoaderError)
        assert issubclass(DocumentParseError, DocumentLoaderError)


class TestLoadFileErrorHandling:
    def test_unsupported_extension_raises_unsupported_file_type(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(UnsupportedFileTypeError) as exc:
            load_file(str(f))
        assert ".xyz" in str(exc.value)

    def test_txt_file_loads_successfully(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        docs = load_file(str(f))
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"
