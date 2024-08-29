import matplotlib.pyplot as plt

def plot_data(X_train, X_test, y_train, y_test):
    plt.style.use('fivethirtyeight')
    plt.scatter(X_train, y_train, marker = 'x', c = 'r', label = 'training')
    plt.scatter(X_test, y_test, marker = 'o', c = 'b', label = 'test')
    plt.title('Input vs Target')
    plt.xlabel('Years Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_model(X_train, X_test, y_train, y_test,X_range, y_range_1, y_range_2, model_name_1, model_name_2):
    plt.style.use('fivethirtyeight')
    
    fig, ax = plt.subplots(1, 2, figsize = (16,4))
    ax[0].scatter(X_train, y_train, marker = 'x', c = 'r', label = 'training')
    ax[0].scatter(X_test, y_test, marker = 'o', c = 'b', label = 'test')
    ax[0].plot(X_train, y_range_1, c = 'g', label = 'model')
    ax[0].set_title(model_name_1)
    ax[0].set_xlabel('Years Experience')
    ax[0].set_ylabel('Salary')
    ax[0].legend()
    
    ax[1].scatter(X_train, y_train, marker = 'x', c = 'r', label = 'training')
    ax[1].scatter(X_test, y_test, marker = 'o', c = 'b', label = 'test')
    ax[1].plot(X_range, y_range_2, c = 'g', label = 'model')
    ax[1].set_title(model_name_2)
    ax[1].set_xlabel('YearsExperience')
    ax[0].set_ylabel('Salary')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()