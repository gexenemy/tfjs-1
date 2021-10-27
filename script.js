import {MnistData} from './data.js';

async function showExamples(data) {
    // Создает visor
    const surface =
        tfvis.visor().surface({name: 'Input Data Examples', tab: 'Input Data'});

    // Получает примеры
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Создайте элемент холста для рендеринга каждого примера
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Измените форму изображения на 28x28 пикселей.
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // В первом слое нашей сверточной нейронной сети мы должны указать форму ввода.
    // Затем мы указываем некоторые параметры для операции свертки, которая происходит на этом слое.
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        // inputShape. Форма данных, которые будут перетекать в первый слой модели. В данном случае наши примеры MNIST представляют собой черно-белые изображения размером 28x28 пикселей. Канонический формат для данных изображения [row, column, depth], поэтому здесь мы хотим настроить форму [28, 28, 1]. 28 строк и столбцов для количества пикселей в каждом измерении и глубина 1, потому что наши изображения имеют только 1 цветовой канал. Обратите внимание, что мы не указываем размер партии во входной форме. Слои разработаны так, чтобы не зависеть от размера пакета, поэтому во время вывода вы можете передать тензор любого размера пакета.
        kernelSize: 5,
        // kernelSize. Размер окон скользящего сверточного фильтра, применяемых к входным данным. Здесь мы устанавливаем kernelSizeof 5, которое задает квадратное сверточное окно 5x5.
        filters: 8,
        // filters. Количество окон фильтра, размер которых kernelSize применяется к входным данным. Здесь мы применим к данным 8 фильтров.
        strides: 1,
        // strides. «Размер шага» скользящего окна, то есть на сколько пикселей фильтр будет сдвигать каждый раз, когда перемещается по изображению. Здесь мы указываем шаг 1, что означает, что фильтр будет скользить по изображению с шагом в 1 пиксель.
        activation: 'relu',
        // activation. Функция активации, применяемая к данным после завершения свертки. В этом случае мы применяем функцию выпрямленных линейных единиц (ReLU) , которая является очень распространенной функцией активации в моделях машинного обучения.
        kernelInitializer: 'varianceScaling'
        // kernelInitializer. Метод, используемый для случайной инициализации весов модели, что очень важно для динамики обучения.
    }));

    // Слой MaxPooling действует как своего рода понижающая дискретизация с использованием максимальных значений в регионе вместо усреднения.
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Повторите другой стек conv2d + maxPooling. Обратите внимание, что в свертке больше фильтров.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Теперь мы сглаживаем выходной сигнал 2D-фильтров в одномерный вектор, чтобы подготовить его для ввода в наш последний слой.
    // Это обычная практика при подаче данных более высокого уровня в выходной слой окончательной классификации.
    model.add(tf.layers.flatten());

    // Наш последний слой - это плотный слой, который имеет 10 единиц вывода, по одной для каждого класса вывода (то есть 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    // Выберите оптимизатор, функцию потерь и метрику точности, затем скомпилируйте и верните модель.
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
    await train(model, data);
    await showAccuracy(model, data);
    await showConfusion(model, data)
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    // Здесь мы делаем два набора данных, обучающий набор, на котором мы будем обучать модель, и набор проверки,
    // на котором мы будем тестировать модель в конце каждой эпохи,
    // однако данные в наборе проверки никогда не отображаются для модели во время обучения. .
    // Предоставленный нами класс данных упрощает получение тензоров из данных изображения.
    // Но мы по-прежнему преобразуем тензоры в форму, ожидаемую моделью, [num_examples, image_width, image_height, channels],
    // прежде чем мы сможем передать их модели. Для каждого набора данных у нас есть как входные данные (X), так и метки (Y).

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    // Мы вызываем model.fit, чтобы начать цикл обучения. Мы также передаем свойство validationData,
    // чтобы указать, какие данные модель должна использовать для тестирования себя после каждой эпохи (но не использовать для обучения).
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

// Для начала нам нужно сделать некоторые прогнозы. Здесь мы возьмем 500 изображений и предскажем, какая цифра в них.
// Примечательно, что argmaxфункция - это то, что дает нам индекс наивысшего класса вероятности.
// Помните, что модель выводит вероятность для каждого класса.
// Здесь мы определяем наивысшую вероятность и назначаем ее использование в качестве прогноза.
// doPredictions показывает, как вы обычно можете делать прогнозы после обучения вашей модели. Однако с новыми данными у вас не будет существующих ярлыков.
function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}

// С помощью набора прогнозов и меток мы можем рассчитать точность для каждого класса.
async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {name: 'Accuracy', tab: 'Evaluation'};
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

// Матрица неточностей похожа на точность для каждого класса, но дополнительно разбивает ее, чтобы показать образцы неправильной классификации. Это позволяет увидеть, не запуталась ли модель в каких-либо конкретных парах классов.
async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

    labels.dispose();
}

document.addEventListener('DOMContentLoaded', run);

