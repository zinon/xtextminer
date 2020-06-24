from sklearn.feature_extraction.text import TfidfTransformer

custom_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
