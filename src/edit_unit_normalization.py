"Script to edit the unit normalization dictionary CSV file."

import pandas as pd


class UnitNormalizationDict:
    """Class to manage the unit normalization dictionary."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.unit_dict = pd.read_csv(self.file_path)

    def add_new_entry(self, entry: dict):
        """Add a new entry to the unit normalization dictionary."""
        key_name = entry['unit']
        value = entry['normalized_unit']
        # Default factor to 1.0 if not provided
        factor = entry.get('factor', 1.0)
        if key_name not in self.unit_dict['unit'].values:
            new_entry = pd.DataFrame(
                {'unit': [key_name], 'normalized_unit': [value], 'factor': [factor]})
            self.unit_dict = pd.concat(
                [self.unit_dict, new_entry], ignore_index=True)

    def remove_entry(self, key_name: str):
        """Remove an entry from the unit normalization dictionary."""
        self.unit_dict = self.unit_dict[self.unit_dict['unit'] != key_name]

    def save(self):
        """Save the updated unit normalization dictionary to CSV."""
        self.unit_dict.to_csv(self.file_path, index=False)


if __name__ == "__main__":
    unit_normalization_dict = UnitNormalizationDict(
        "data/normalization_units/unit_normalization_dict.csv")

    # Remove
    unit_normalization_dict.remove_entry(
        key_name="Nothing extracted. No Regex match")
    unit_normalization_dict.remove_entry(
        key_name="Not specified")
    unit_normalization_dict.remove_entry(key_name="<unit not specified>")

    # Add
    entry = {
        'unit': 'million tonnes/year',
        'normalized_unit': 'Mt CO2e',
        'factor': 1000000.0
    }
    unit_normalization_dict.add_new_entry(entry)
    unit_normalization_dict.save()


    # Save changes
    unit_normalization_dict.save()
