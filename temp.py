class my_train(DetectionTrainer):
    def __init__(self,model,trainer,img):
        self.model = model
        self.trainer = trainer
        self.img = img

    def get_loss(self, model, img,trainer):
        results = model.predict(img)
        loss = model.loss(results)
        results = self.model.predict(img)
        arr = results[0].plot()
        im = Image.fromarray(arr)
        im.save("dataset/train/your_file.jpeg")
        im.save("dataset/train/your_file1.jpeg")
        tr = self.model.train(epochs=2,batch=1,data='data.yaml',plots=False)
        print(tr)