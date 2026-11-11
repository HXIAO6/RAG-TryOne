class Document:

    def __init__(self, name, page):
        self.name = name
        self.page = page

    def show(self):
        print("文件名:", self.name)
        print("页码:", self.page)


doc1 = Document("manual.pdf", 1)
doc2 = Document("manual.pdf", 2)

doc1.show()
doc2.show()