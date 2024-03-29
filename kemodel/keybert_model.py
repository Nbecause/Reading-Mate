class KeywordExtraction():
    def __init__(self, kw_model, num_to_gen, stop_words, use_maxsum, nr_candidates, use_mmr, diversity):
        self.kw_model = kw_model
        self.num_to_gen = num_to_gen
        self.stop_words = stop_words
        self.use_maxsum = use_maxsum
        self.nr_candidates = nr_candidates
        self.use_mmr = use_mmr
        self.diversity = diversity
        
    def generate_keywords(self, context, n_gram):
        """ Generate keywords

        Args:
            context (string): Context to extract keywords
            n_gram (int): Number of adjacent characters

        Returns:
            keywords (list): extracted keywords
        """
        if self.use_maxsum == 'False' and self.use_mmr == 'False':
            keywords_candidates = self.kw_model.extract_keywords(context, 
                                        keyphrase_ngram_range=(1, n_gram), 
                                        stop_words=self.stop_words, 
                                        top_n=self.num_to_gen,
                                        )
                    
        elif self.use_maxsum == 'True' and self.use_mmr == 'False':
            keywords_candidates = self.kw_model.extract_keywords(context, 
                                        keyphrase_ngram_range=(1, n_gram), 
                                        stop_words=self.stop_words, 
                                        top_n=self.num_to_gen,
                                        use_maxsum=self.use_maxsum,
                                        nr_candidates=self.nr_candidates,
                                        )
        elif self.use_maxsum == 'False' and self.use_mmr == 'True':
            keywords_candidates = self.kw_model.extract_keywords(context, 
                                        keyphrase_ngram_range=(1, n_gram), 
                                        stop_words=self.stop_words, 
                                        top_n=self.num_to_gen,
                                        use_mmr=self.use_mmr,
                                        diversity=self.diversity)
            
        elif self.use_maxsum == 'True' and self.use_mmr == 'True':
            keywords_candidates = self.kw_model.extract_keywords(context, 
                                        keyphrase_ngram_range=(1, n_gram), 
                                        stop_words=self.stop_words, 
                                        top_n=self.num_to_gen,
                                        use_maxsum=self.use_maxsum,
                                        nr_candidates=self.nr_candidates,
                                        use_mmr=self.use_mmr,
                                        diversity=self.diversity)
            
        keywords = [k[0] for k in keywords_candidates]

        return keywords