import re


def parse_deep_seek(s):
    """
    Parse a string containing '<think>' ... '</think>' sections.

    Args:
        s (str): Input string.

    Returns:
        dict: A dictionary where keys are 'answer' and values are the rest 
              of the input string,
              and key 'think' with value is the content within '<think> 
              ... '</think>' sections.
    """
    # Regular expression pattern to match '<think>' ... '</think>' sections
    pattern = r'<think>(.*?)</think>'
    think_content = re.findall(pattern, s)
    answer = re.sub(pattern, '', s)
    return {'answer': answer.strip(), 'think': think_content[0] if think_content else ''}





def split_stream_into_speech_chunks(stream_generator, punctuation_chars={".", "!", "?", ":", "\n"}):
    """
    Splits a stream of words into speech chunks based on a set of punctuation characters.
    Yields complete chunks ending in a punctuation character.
    """
    buffer = ""
    # Escape punctuation for regex
    punctuation_regex = "|".join(re.escape(p) for p in punctuation_chars)

    # Regex pattern for sentence-ending punctuation
    sentence_end_pattern = re.compile(rf'(.*?[{punctuation_regex}])(\s|$)')

    for word in stream_generator:
        buffer += word

        # Look for any complete chunk
        while True:
            match = sentence_end_pattern.search(buffer)
            if match:
                chunk = match.group(1).strip()
                yield chunk
                buffer = buffer[match.end():]
            else:
                break

    # Yield any leftover buffer as a final chunk
    if buffer.strip():
        yield buffer.strip()






