import os
import matplotlib.pyplot as plt
from source.utils.utils import make_dir

class History:
    def __init__(self, target_type_string):
        self.target_type_string = target_type_string

        if target_type_string == 'Classification':
            self.history = {
                'train_loss': [],
                'train_accuracy': [],
                'train_recall': [],
                'train_precision': [],
                'train_f1': [],

                'valid_loss': [],
                'valid_accuracy': [],
                'valid_recall': [],
                'valid_precision': [],
                'valid_f1': [],
                
                'training_time': []
            }
        elif target_type_string == 'Regression':
            self.history = {
                'train_loss': [],
                'train_r2': [],
                'train_mae': [],
                'train_mse': [],

                'valid_loss': [],
                'valid_r2': [],
                'valid_mae': [],
                'valid_mse': [],
                
                'training_time': []
            }

    def save_step(self, train_loss, valid_loss, train_metrics, valid_metrics, training_time):
        self.history['train_loss'].append(train_loss)
        self.history['valid_loss'].append(valid_loss)
        self.history['training_time'].append(training_time)

        if self.target_type_string == 'Classification':
            self.history['train_accuracy'].append(train_metrics.accuracy)
            self.history['train_recall'].append(train_metrics.recall)
            self.history['train_precision'].append(train_metrics.precision)
            self.history['train_f1'].append(train_metrics.f1)

            self.history['valid_accuracy'].append(valid_metrics.accuracy)
            self.history['valid_recall'].append(valid_metrics.recall)
            self.history['valid_precision'].append(valid_metrics.precision)
            self.history['valid_f1'].append(valid_metrics.f1)

        if self.target_type_string == 'Regression':
            self.history['train_r2'].append(train_metrics.r2_score)
            self.history['train_mae'].append(train_metrics.mean_abs_error)
            self.history['train_mse'].append(train_metrics.mean_square_error)

            self.history['valid_r2'].append(train_metrics.r2_score)
            self.history['valid_mae'].append(train_metrics.mean_abs_error)
            self.history['valid_mse'].append(train_metrics.mean_square_error)

    def _base_display(self, path_string, file_string):
        epoch = len(self.history['train_loss'])
        epochs = list(range(1, epoch + 1))
        plt.xticks(epochs)

        fig, axes = plt.subplots(3, 1)

        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['valid_loss'], label='Validation')
        axes[0].legend()

        if self.target_type_string == 'Classification':
            axes[1].set_title('Accuracy Score')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].plot(epochs, self.history['train_accuracy'], label='Train')
            axes[1].plot(epochs, self.history['valid_accuracy'], label='Validation')
            axes[1].legend()

            axes[2].set_title('F1 Score (Beta = 1)')
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('F1')
            axes[2].plot(epochs, self.history['train_f1'], label='Train')
            axes[2].plot(epochs, self.history['valid_f1'], label='Validation')
            axes[2].legend()
			
        elif self.target_type_string == 'Regression':
            axes[1].set_title('R2 Score')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('R2')
            axes[1].plot(epochs, self.history['train_r2'], label='Train')
            axes[1].plot(epochs, self.history['valid_r2'], label='Validation')
            axes[1].legend()

            axes[2].set_title('MSE Score')
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('MSE')
            axes[2].plot(epochs, self.history['train_mse'], label='Train')
            axes[2].plot(epochs, self.history['valid_mse'], label='Validation')
            axes[2].legend()

        plt.tight_layout()
        img_extension = '.png'
        base_name = '_history'
        folder_path = make_dir(os.path.join('.', 'plots', path_string))
        save_string = os.path.join(folder_path, file_string + base_name + img_extension)
        fig.savefig(save_string)

    def history_display(self, path_string, file_string, type_display='base'):
        if type_display is None or type_display == 'base':
            self._base_display(path_string, file_string)
        else:
            self._base_display(path_string, file_string)