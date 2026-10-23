# /// script
# dependencies = []
# ///
input_1: ANY = None
input_2: ANY = None
input_3: ANY = None
input_4: ANY = None
input_5: ANY = None
choices: str = "Input 1|Input 2|Input 3|Input 4|Input 5"
selected: int = 0

values = [input_1, input_2, input_3, input_4, input_5]
labels = [label.strip() for label in str(choices).split("|")]
while len(labels) < 5:
    labels.append(f"Input {len(labels) + 1}")
labels = labels[:5]
selected_index = int(selected)
if selected_index < 0 or selected_index >= 5:
    raise ValueError("Switch selection must be between 0 and 4")
chosen = values[selected_index]
if chosen is None:
    raise ValueError(f"Switch input '{labels[selected_index]}' is not connected")

return chosen: ANY
