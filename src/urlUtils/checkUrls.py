import pandas as pd
from pathlib import Path
import os

# The approximated URL mapping we generated
NEW_TOPIC_MAPPING = {
    '/lektionen/grundlagen/': [
        '/themen/einfuehrung/',
        '/themen/begriffe/',
        '/themen/schriftliche-addition/',
        '/tests/schriftliche-addition-mit-kommas/',
        '/themen/schriftliche-subtraktion/',
        '/themen/schriftliche-multiplikation/',
        '/tests/schriftliche-multiplikation-mit-kommazahlen/',
        '/themen/schriftliche-division/',
        '/tests/schriftliche-division/',
        '/themen/bruchrechnen/',
        '/themen/masseinheiten/',
        '/tests/masseinheiten-aufgabenset-1/',
        '/tests/masseinheiten-aufgabenset-2/',
        '/themen/7-gemischte-aufgaben/',
        '/tests/gemischte-rechnungen-aufgabenset-1/',
        '/tests/gemischte-rechnungen-aufgabenset-2/',
    ],
    '/lektionen/2-proportionalitaet/': [
        '/themen/0-einfuehrung/',
        '/themen/1-direkte-proportionalitaet/',
        '/themen/2-indirekte-proportionalitaet/',
        '/tests/aufgabenset-1-proportionalitaeten/',
        '/tests/proportionalitaeten-aufgabenset-2/',
        '/themen/3-dreisatz/',
        '/tests/sfwd-quiz-634aa0b3c83121-33375968/',
        '/themen/zusammenfassung/',
        '/tests/proportionalitaten-prufungsaufgaben-set-1/',
        '/tests/proportionalitaten-pruefungsaufgaben-set-2/',
    ],
    '/lektionen/3-gleichungen/': [
        '/themen/1-einfuehrung/',
        '/themen/variablen-und-terme/',
        '/themen/gleichungen-loesen/',
        '/tests/gleichungen-aufgabenset-1/',
        '/tests/gleichungen-aufgabenset-2/',
        '/tests/gleichungen-aufgabenset-3/',
        '/themen/weg-zeit-und-tempo/',
        '/tests/geschwindigkeit-aufgabenset-1/',
        '/themen/zaeune-und-pfosten/',
        '/tests/zaeune-pfosten-aufgabenset-1/',
        '/themen/4-abfluss-und-zufluss/',
        '/tests/ab-und-zufluss-aufgabenset-1/',
        '/themen/5-planaenderungsaufgaben/',
        '/tests/plananderungsaufgaben-aufgabenset-1/',
    ],
    '/lektionen/4-kombinatorik-und-zahlenraetsel/': [
        '/themen/einfuehrung-kombinatorik/',
        '/themen/das-additionsprinzip/',
        '/themen/das-multiplikationsprinzip/',
        '/themen/kombinatorik-und-zahlenraetsel/',
        '/themen/zahlenraetsel/',
        '/tests/zahlenratsel-aufgabenset-1/',
        '/tests/zahlenraetsel-aufgabenset-2/',
        '/tests/zahlenraetsel-aufgabenset-3/',
    ],
    '/lektionen/5-konstruktionen-strecken-und-umfang/': [
        '/themen/einfuehrung-5/',
        '/themen/geometrische-transformationen/',
        '/themen/konstruktionen-mit-strecken-winkeln-und-der-mittelsenkrechten/',
        '/themen/ebenen-der-geometrie/',
        '/themen/3-umfang-und-streckeberechnungen/',
        '/themen/volumenberechnung-von-quadern/',
        '/tests/umfang-aufgabenset-1/',
        '/tests/streckenberechnung-aufgabenset-2/',
        '/tests/umfang-pruefungsaufgaben/',
    ],
    '/lektionen/6-vorstellungsvermoegen/': [
        '/themen/raumvorstellungsaufgaben-in-der-geometrie/',
        '/themen/aufsicht-vorderansicht-seitenansicht-und-schraegbilder/',
        '/themen/arbeiten-und-zeichnen-mit-netzen-von-wuerfeln-und-quadern/',
    ],
    '/lektionen/uebungspruefungen-und-teststrategien-2/': [
        '/themen/einfuehrung-in-teststrategien-2/',
        '/themen/uebungspruefung-1-und-besprechung-2/',
        '/themen/vertiefung-in-ausgewaehlte-teststrategien-2/',
        '/themen/uebungspruefung-2-und-besprechung-2/',
        '/themen/abschluss-und-letzte-tipps-2/',
    ],
    '/lektionen/langzeit-mathe-pruefungsmodul/': [
        '/themen/2025-prufung-zap1/',
        '/themen/2024-pruefung-zap1/',
        '/themen/2023-pruefung-zap1/',
        '/themen/2022-pruefung-zap1/',
        '/themen/2021-pruefung-zap1/',
        '/themen/2020-pruefung-zap1/',
        '/themen/2019-pruefung-zap1/',
        '/themen/2018-pruefung-zap1/',
        '/themen/2017-pruefung-zap1/',
        '/themen/2016-pruefung-zap1/',
        '/themen/2015-pruefung-zap1/',
        '/themen/2014-pruefung-zap1/',
        '/themen/2013-pruefung-zap1/',
    ],
}

def get_expected_urls(mapping):
    """Extracts all module (keys) and lesson/quiz (values) URLs into a single set."""
    expected_urls = set()
    for module_url, lesson_urls in mapping.items():
        expected_urls.add(module_url)
        expected_urls.update(lesson_urls)
    return expected_urls

def check_missing_urls(csv_file_path):
    # 1. Gather all the expected URLs
    expected_urls = get_expected_urls(NEW_TOPIC_MAPPING)
    
    try:
        # 2. Load the CSV file
        print(f"Loading '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)
        
        # Ensure the 'url' column exists
        if 'url' not in df.columns:
            print("Error: The CSV file does not contain a 'url' column.")
            return

        # 3. Extract unique URLs from the CSV
        # Using dropna() to ignore empty rows, and set() for fast lookups
        visited_urls = set(df['url'].dropna().unique())
        
        # 4. Find the difference: URLs we expect but didn't find in the CSV
        missing_urls = expected_urls - visited_urls
        
        # 5. Output the results
        print("-" * 40)
        if missing_urls:
            print(f"Found {len(missing_urls)} missing URLs that do not appear in the CSV:\n")
            # Print them sorted alphabetically for easier reading
            for url in sorted(missing_urls):
                print(url)
        else:
            print("Great! All deduced URLs appear in your pageviews.csv file.")
            
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' could not be found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure pageviews.csv is in the same directory as this script, or update the path
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_PATH   = os.path.join(ROOT_DIR, "data/pageviews.csv")
    check_missing_urls(RESULTS_PATH)