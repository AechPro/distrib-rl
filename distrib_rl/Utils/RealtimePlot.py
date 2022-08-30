import threading
import matplotlib.pyplot as plt


class RealtimePlot(threading.Thread):
    def __init__(self, interval):
        super().__init__()
        self.data = []
        self.axis = None
        self.interval = interval
        self.key = threading.Lock()
        self.running = False

    def run(self):
        top = 125
        bottom = 0
        left = -400
        right = 150
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_title("Value Estimator Histogram")
        ax.set_xlabel("Discounted Future Reward")
        ax.set_ylabel("Count")
        ax.set_ylim(top=top)
        ax.set_xlim(left=left, right=right)
        ax.legend(["pred", "true"])

        fig.canvas.draw()
        plt.show()
        self.running = True

        while self.running:
            if self.data is not None and len(self.data) > 0:
                with self.key:
                    ax.clear()
                    ax.set_title("Value Estimator Histogram")
                    ax.set_xlabel("Discounted Future Reward")
                    ax.set_ylabel("Count")
                    ax.set_ylim(top=top)
                    ax.set_xlim(left=left, right=right)
                    ax.legend(["pred", "true"])

                    pred, true = self.data
                    _, _, bar_container = ax.hist(true, bins=100, lw=1, alpha=0.5)
                    _, _, bar_container = ax.hist(pred, bins=100, lw=1, alpha=0.5)
                    fig.canvas.draw()

            plt.pause(self.interval)

    def stop(self):
        self.running = False

    def pass_data(self, data):
        with self.key:
            self.data = data
