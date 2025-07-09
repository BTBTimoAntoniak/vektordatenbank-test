import json

class TechnologieWissen:
    def __init__(self, name: str, letzte_verwendung: int):
        self.name = name
        self.letzte_verwendung = letzte_verwendung

    def __repr__(self):
        return f"{self.name}: vor {self.letzte_verwendung} Tagen"

class MitarbeiterSkills:
    def __init__(self, name: str, technologien: list[TechnologieWissen]):
        self.name = name
        self.technologien = technologien

    def __repr__(self):
        technologien_str = "\n    ".join([repr(t) for t in self.technologien])
        return (
            f"Mitarbeiter:\n"
            f"  Name: {self.name}\n"
            f"  Skills:\n"
            f"    {technologien_str}"
        )
    
    def to_embedding(self):
        # Konvertiere die Technologien in eine Liste von Strings
        technologien_str = [t.name for t in self.technologien]
        # Erstelle ein Embedding aus den Technologien
        return " ".join(technologien_str)
    
def get_test_data() -> list[MitarbeiterSkills]:
    """
    Liest die Daten aus der Datei 'mitarbeiter_skills.json' und gibt eine Liste von MitarbeiterSkills-Objekten zurück.
    """
    with open('mitarbeiter_skills.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    mitarbeiter_list = []
    for mitarbeiter in data:
        name = mitarbeiter.get('name', '')
        technologien = []
        for tech in mitarbeiter.get('technologien', []):
            tech_name = tech.get('name', None)
            letzte_verwendung = tech.get('letzte_verwendung', None)
            if tech_name is not None and letzte_verwendung is not None:
                technologien.append(TechnologieWissen(tech_name, letzte_verwendung))
        mitarbeiter_list.append(MitarbeiterSkills(name, technologien))
    return mitarbeiter_list

