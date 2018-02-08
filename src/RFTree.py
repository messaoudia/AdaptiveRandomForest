import skmultiflow as skm

class RFTree(skm.HoeffdingTree):

    def update(self, m, x, y):
        print("This function will update the tree")