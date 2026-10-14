import os
import sys
import io
import zipfile

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


def download_and_extract_jvs():
    """
    Tải và giải nén JVS Corpus (100 người Nhật bản xứ, ~30 giờ âm thanh sạch, ~3.54 GB)
    """
    target_dir = os.path.abspath("jvs_corpus")
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "jvs_ver1.zip")
    extracted_root = os.path.join(target_dir, "jvs_ver1")

    # Dọn dẹp file .part nếu có
    for f in os.listdir(target_dir):
        if f.endswith(".part"):
            try:
                os.remove(os.path.join(target_dir, f))
            except Exception:
                pass

    # 1. Kiểm tra file zip đã tải đủ chưa
    zip_ready = False
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 3_000_000_000:
        try:
            with zipfile.ZipFile(zip_path, "r") as test_zf:
                if test_zf.testzip() is None:
                    zip_ready = True
        except Exception:
            zip_ready = False

    if zip_ready:
        print("=" * 70)
        print("✅ FILE ZIP JVS CORPUS ĐÃ TẢI XONG ĐỦ 100% (~3.54 GB) TRÊN MÁY!")
        print("=" * 70)
    else:
        try:
            import gdown
        except ImportError:
            print("[!] Chưa cài đặt thư viện 'gdown'. Hãy chạy: pip install gdown")
            sys.exit(1)

        file_id = "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"

        print("=" * 70)
        print(">>> [1/2] ĐANG TẢI JVS CORPUS TỪ GOOGLE DRIVE (~3.54 GB)...")
        print("=" * 70)
        gdown.download(gdrive_url, zip_path, quiet=False)

    # 2. Tiến hành giải nén
    print("\n" + "=" * 70)
    print(">>> [2/2] ĐANG TIẾN HÀNH GIẢI NÉN FILE jvs_ver1.zip (42,372 files)...")
    print(">>> Vui lòng đợi trong giây lát...")
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    print("\n[+] Giải nén thành công!")

    # 3. Kiểm tra số lượng speaker
    if os.path.exists(extracted_root):
        speakers = [
            d for d in os.listdir(extracted_root)
            if os.path.isdir(os.path.join(extracted_root, d)) and d.startswith("jvs")
        ]
        print(f"🎉 HOÀN TẤT 100%! Đã giải nén đủ {len(speakers)} thư mục speaker (từ jvs001 đến jvs100).")
        print(f"📁 Vị trí dữ liệu sẵn sàng: {extracted_root}")

    print("\n(Mẹo: Bạn có thể xóa file nén 'jvs_corpus/jvs_ver1.zip' nếu muốn giải phóng 3.54 GB ổ đĩa).")


if __name__ == "__main__":
    download_and_extract_jvs()
