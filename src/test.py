

class Test(self):
    def __init__(self):
        return self
    
    def decode_integers_to_categorical(self, arr, mapping):
        results = []
        for item in arr:
            x = mapping.get(item)
            results.append(x)
        return results
