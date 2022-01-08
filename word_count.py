import io

from nbformat import current

# filepath="/content/drive/My Drive/NLE Notebooks/assessment/assignment1.ipynb"

filepath = "/home/sahil/dsrep/dsrep.ipynb"

question_count = 437


with io.open(filepath, "r", encoding="utf-8") as f:

    nb = current.read(f, "json")


word_count = 0

for cell in nb.worksheets[0].cells:

    if cell.cell_type == "markdown":

        word_count += len(cell["source"].replace("#", "").lstrip().split(" "))

print("Submission length is {}".format(word_count))
