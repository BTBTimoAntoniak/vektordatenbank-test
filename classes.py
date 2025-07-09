import json
from typing import List

class TechnologieWissen:
    def __init__(self, name: str, letzte_verwendung: int):
        self.name = name
        self.letzte_verwendung = letzte_verwendung

    def __repr__(self):
        return f"{self.name}: vor {self.letzte_verwendung} Tagen"

class MitarbeiterSkills:
    def __init__(self, name: str, technologien: List[TechnologieWissen]):
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
    
    def __str__(self):
        technologien_str = ", ".join([t.name for t in self.technologien])
        return f"{self.name} ({technologien_str})"
    
    def to_embedding(self) -> str:
        """Konvertiert die Technologien in einen Fließtext für das Embedding."""
        technologien_str = [t.name for t in self.technologien]
        return " ".join(technologien_str)

    @staticmethod
    def from_dict(data: dict) -> 'MitarbeiterSkills':
        name = data.get('name', '')
        technologien = [
            TechnologieWissen(tech.get('name', ''), tech.get('letzte_verwendung', 0))
            for tech in data.get('technologien', [])
        ]
        return MitarbeiterSkills(name, technologien)

def get_test_data() -> List[MitarbeiterSkills]:
    """
    Liest die Daten aus der Datei 'mitarbeiter_skills.json' und gibt eine Liste von MitarbeiterSkills-Objekten zurück.
    """
    with open('mitarbeiter_skills.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [MitarbeiterSkills.from_dict(mitarbeiter) for mitarbeiter in data]

