"""
Egyptian National ID number: generation and structural validation.

Format (14 digits): C YYMMDD GG SSSS Z
  C      - century digit: 2 -> born 1900-1999, 3 -> born 2000-2099
  YYMMDD - date of birth
  GG     - 2-digit governorate code of birth/registration
  SSSS   - 4-digit sequence number for that day/governorate
  Z      - check/parity digit (odd = male, even = female by convention;
           we don't implement the real Ministry of Interior checksum
           algorithm since it isn't publicly documented in full -
           we use a simple deterministic check digit instead, which is
           enough for our purposes: testing that downstream regex/format
           validation in the OCR pipeline works correctly)

NOTE: This module produces structurally-valid-looking IDs for synthetic
training/test data only. It does not implement the real government
checksum and must never be used to validate or fabricate real ID numbers.
"""
import random

GOVERNORATE_CODES = {
    "01": "القاهرة",       # Cairo
    "02": "الإسكندرية",     # Alexandria
    "03": "بورسعيد",        # Port Said
    "04": "السويس",         # Suez
    "11": "دمياط",          # Damietta
    "12": "الدقهلية",       # Dakahlia
    "13": "الشرقية",        # Sharqia
    "14": "القليوبية",      # Qalyubia
    "15": "كفر الشيخ",      # Kafr El Sheikh
    "16": "الغربية",        # Gharbia
    "17": "المنوفية",       # Monufia
    "18": "البحيرة",        # Beheira
    "19": "الإسماعيلية",    # Ismailia
    "21": "الجيزة",         # Giza
    "22": "بني سويف",       # Beni Suef
    "23": "الفيوم",         # Fayoum
    "24": "المنيا",         # Minya
    "25": "أسيوط",          # Assiut
    "26": "سوهاج",          # Sohag
    "27": "قنا",            # Qena
    "28": "أسوان",          # Aswan
    "29": "الأقصر",         # Luxor
    "31": "البحر الأحمر",   # Red Sea
    "32": "الوادي الجديد",  # New Valley
    "33": "مطروح",          # Matrouh
    "34": "شمال سيناء",     # North Sinai
    "35": "جنوب سيناء",     # South Sinai
}


def _check_digit(digits_13: str) -> str:
    """
    Deterministic check digit (NOT the real government algorithm).
    Weighted sum mod 10, just to give our synthetic IDs an internally
    consistent, regex-and-logic-testable 14th digit.
    """
    total = sum(int(d) * (i + 2) for i, d in enumerate(digits_13))
    return str(total % 10)


def generate_national_id(birth_year: int, birth_month: int, birth_day: int,
                          governorate_code: str, sequence: int, is_male: bool) -> str:
    century_digit = "2" if 1900 <= birth_year <= 1999 else "3"
    yy = f"{birth_year % 100:02d}"
    mm = f"{birth_month:02d}"
    dd = f"{birth_day:02d}"
    seq = f"{sequence:04d}"

    # last digit of the 4-digit sequence carries gender parity by convention
    parity_base = int(seq[3])
    if is_male and parity_base % 2 == 0:
        parity_base = (parity_base + 1) % 10
    if not is_male and parity_base % 2 == 1:
        parity_base = (parity_base + 1) % 10
    seq = seq[:3] + str(parity_base)

    first_13 = f"{century_digit}{yy}{mm}{dd}{governorate_code}{seq}"
    assert len(first_13) == 13, f"expected 13 digits before check, got {len(first_13)}"

    check = _check_digit(first_13)
    return first_13 + check


def random_national_id(rng: random.Random) -> dict:
    is_male = rng.random() < 0.5
    birth_year = rng.randint(1970, 2005)
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    gov_code = rng.choice(list(GOVERNORATE_CODES.keys()))
    sequence = rng.randint(0, 999)

    nid = generate_national_id(birth_year, birth_month, birth_day,
                                gov_code, sequence, is_male)
    return {
        "national_id": nid,
        "birth_year": birth_year,
        "birth_month": birth_month,
        "birth_day": birth_day,
        "governorate_code": gov_code,
        "governorate_name": GOVERNORATE_CODES[gov_code],
        "is_male": is_male,
    }


def western_to_eastern_arabic_numerals(text: str) -> str:
    """Convert 0-9 to Eastern Arabic-Indic digits ٠-٩."""
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    table = str.maketrans(western, eastern)
    return text.translate(table)


def eastern_to_western_arabic_numerals(text: str) -> str:
    """Convert Eastern Arabic-Indic digits ٠-٩ back to 0-9."""
    eastern = "٠١٢٣٤٥٦٧٨٩"
    western = "0123456789"
    table = str.maketrans(eastern, western)
    return text.translate(table)


if __name__ == "__main__":
    rng = random.Random(42)
    for _ in range(5):
        info = random_national_id(rng)
        print(info["national_id"], "->", info["governorate_name"],
              "male" if info["is_male"] else "female")
        print("  eastern form:", western_to_eastern_arabic_numerals(info["national_id"]))
