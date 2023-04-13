import json
class Cookie():
    def __init__(self):
        f = open('cookie.json')
        self.data = json.load(f)
        f.close()

    def set(self,key,value):
        self.data[key] = value 
    def get(self, key ):
        if key not in self.data.keys():
            return None
        return self.data[key]
    def delete(self , key):
        self.data.pop(key,None)
    def __del__(self):
        json_object = json.dumps(self.data, indent=4)
 
        # Writing to sample.json
        with open("cookie.json", "w") as outfile:
            outfile.write(json_object)