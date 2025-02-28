
import numpy as np
from textsplit.tools import SimpleSentenceTokenizer, get_segments

tokenizer = SimpleSentenceTokenizer()

def text_splitter(text, max_sentences_per_chunk=42):
    """
    Splits text into coherent chunks using SimpleSentenceTokenizer and a fixed
    number of sentences per chunk.

    Args:
        text (str): The input text to be split.
        max_sentences_per_chunk (int): Maximum number of sentences in each chunk.

    Returns:
        list of str: List of coherent text chunks.
    """
    # Step 1: Tokenize the text into sentences
    sentences = tokenizer(text)

    # Step 2: Define segmentation points
    num_sentences = len(sentences)
    segmentation = np.arange(max_sentences_per_chunk, num_sentences, max_sentences_per_chunk)

    # Step 3: Get coherent chunks
    chunks = get_segments(sentences, segmentation=type('Segmentation', (object,), {'splits': segmentation.tolist()}))

    # Step 4: Combine sentences back into chunks
    return [' '.join(chunk) for chunk in chunks]



if __name__ == "__main__":

	# Example Usage
	text = """
	In the beginning, humans relied on hunting and gathering for survival. Over time, agriculture was developed, which drastically changed human society.
	The industrial revolution brought further changes, with machines taking over much of the manual labor.

	In another part of the world, civilizations like the Indus Valley had complex societal structures and trade networks.
	Technological advancements were slower, but their contributions to urban planning and agriculture were significant.

	Modern technology, such as AI, is revolutionizing industries globally.
	Artificial intelligence has the potential to reshape education, healthcare, and economics.
	"""

	chunks = text_splitter(text, max_sentences_per_chunk=3)

	# Print the chunks
	for i, chunk in enumerate(chunks, 1):
	    print(f"Chunk {i}:\n{chunk}\n")














