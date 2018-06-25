let model = null;

// Load saved model from localStore
loadModel().then((cmodel) => {
    console.log('Se cargo modelo');
    model = cmodel;
});
// End ---

// Async Functions Definitions
async function train(model) {
    const inputs = tf.tensor2d([
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
    ], [16, 2]);
    const results = tf.tensor2d([
        [0],
        [1],
        [2],
        [3],
        [1],
        [2],
        [3],
        [4],
        [2],
        [3],
        [4],
        [5],
        [3],
        [4],
        [5],
        [6],
    ], [16, 1]);

    for (let i = 0; i < 500; i++) {
        const response = await model.fit(inputs, results, {shuffle: true, epochs: 8});
        console.log(response.history.loss[0]);
    }
    await model.save('localstorage://my-model-1');
}

async function loadModel() {
    let cmodel = await tf.loadModel('localstorage://my-model-1');
    if (cmodel === null) {
        model = tf.sequential();
        const hidden = tf.layers.dense({
            units: 3,
            inputShape: [2],
            activation: 'relu'
        });
        const output = tf.layers.dense({
            units: 1,
        });

        model.add(hidden);
        model.add(output);
    }
    cmodel.compile({
        optimizer: tf.train.sgd(0.001),
        loss: 'meanSquaredError'
    });
    return cmodel;
}

// End ---

// Functions for buttons
function btnTrain() {
    train(model).then(() => {
        console.log('Training Complete');
    });
}

function btnCalculate(a, b) {
    let outputs = model.predict(tf.tensor2d([a, b], [1, 2]));
    document.getElementById('result').innerHTML = outputs.toString();
    outputs.print();
}

// End ---