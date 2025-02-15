from database_utils.functions import get_story_wordseqs, lanczosinterp2D

def downsample_vectors(allstories, vectors):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	downsampled_vecs = dict()
	for story in allstories:
		downsampled_vecs[story] = lanczosinterp2D(
		vectors[story], wordseqs[story].data_times,
			wordseqs[story].tr_times, window=3)
	return vectors