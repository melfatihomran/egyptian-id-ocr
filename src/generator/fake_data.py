"""
Pools of common Arabic given names / family names and address components,
used to synthesize plausible-but-fake Egyptian ID card data.

These are common, generic Arabic names/places chosen for realism - they are
NOT drawn from any real individual's data.
"""
import random

MALE_FIRST_NAMES = [
    "محمد", "أحمد", "محمود", "علي", "عمر", "خالد", "إبراهيم", "يوسف",
    "مصطفى", "حسن", "حسين", "كريم", "طارق", "وليد", "عماد", "ياسر",
    "هشام", "سامح", "رامي", "عادل", "فادي", "زياد", "مازن", "أمير",
    "سيف", "عبدالله", "عبدالرحمن", "نور الدين", "بلال", "حمزة",
]

FEMALE_FIRST_NAMES = [
    "فاطمة", "مريم", "نور", "سارة", "ياسمين", "هبة", "منى", "إيمان",
    "دينا", "رانيا", "ريم", "نهى", "سلمى", "آية", "روان", "جنى",
    "ملك", "نادية", "سهر", "وفاء", "أمل", "هدى", "شيماء", "غادة",
]

FAMILY_NAMES = [
    "عبدالله", "محمود", "حسن", "إبراهيم", "السيد", "علي", "أحمد",
    "خليل", "عبدالرحمن", "مصطفى", "الشريف", "النجار", "البدري",
    "حسين", "كامل", "سليمان", "عثمان", "زكي", "فهمي", "راشد",
    "الفاتح", "عمران", "الشيخ", "صادق", "متولي", "عبدالعزيز",
]

# middle name is typically father's first name, grandfather's first name
GRANDFATHER_NAMES = MALE_FIRST_NAMES  # reuse pool, realistic enough

GOVERNORATE_CITIES = {
    "القاهرة": ["مدينة نصر", "المعادي", "حلوان", "مصر الجديدة", "الزمالك", "شبرا"],
    "الجيزة": ["الدقي", "المهندسين", "فيصل", "الهرم", "أكتوبر", "الشيخ زايد"],
    "الإسكندرية": ["سيدي جابر", "محرم بك", "العجمي", "سموحة", "ميامي"],
    "الشرقية": ["الزقازيق", "بلبيس", "العاشر من رمضان", "أبو حماد"],
    "الدقهلية": ["المنصورة", "طلخا", "ميت غمر", "السنبلاوين"],
    "الغربية": ["طنطا", "المحلة الكبرى", "كفر الزيات"],
}

STREET_WORDS = ["شارع", "ميدان", "ممر"]
STREET_NAMES = [
    "النصر", "الجمهورية", "التحرير", "الجيش", "النيل", "الحرية",
    "مكرم عبيد", "عبدالعزيز آل سعود", "الثورة", "الاستقلال", "السلام",
]


def random_full_name(rng: random.Random, is_male: bool) -> str:
    """Egyptian names are typically: First + Father + Grandfather + Family."""
    first = rng.choice(MALE_FIRST_NAMES if is_male else FEMALE_FIRST_NAMES)
    father = rng.choice(GRANDFATHER_NAMES)
    grandfather = rng.choice(GRANDFATHER_NAMES)
    family = rng.choice(FAMILY_NAMES)
    return f"{first} {father} {grandfather} {family}"


def random_address(rng: random.Random) -> str:
    governorate = rng.choice(list(GOVERNORATE_CITIES.keys()))
    city = rng.choice(GOVERNORATE_CITIES[governorate])
    street_word = rng.choice(STREET_WORDS)
    street_name = rng.choice(STREET_NAMES)
    building_no = rng.randint(1, 250)
    return f"{building_no} {street_word} {street_name}، {city}، {governorate}"


if __name__ == "__main__":
    rng = random.Random(7)
    for _ in range(5):
        is_male = rng.random() < 0.5
        print(random_full_name(rng, is_male), "|", random_address(rng))
