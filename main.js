let model = null;

// Load saved model from localStore
loadModel().then((cmodel) => {
    model = cmodel;
});
// End ---

// Async Functions Definitions
async function train(model) {
    const txtoutput = document.getElementById('trainlog');
    txtoutput.innerHTML = "";

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
        [2, 7],
        [6, 4]
    ], [18, 2]);
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
        [9],
        [10]
    ], [18, 1]);

    for (let i = 0; i < 10; i++) {
        const response = await model.fit(inputs, results, {shuffle: true, epochs: 250});
        txtoutput.innerHTML += response.history.loss[0] + '<br>';
        console.log(response.history.loss[0])
    }
    await model.save('localstorage://modelo');
}

async function loadModel() {
    let cmodel = null;

    if (localStorage.getItem("tensorflowjs_models/modelo/info") != null) {
        cmodel = await tf.loadModel('localstorage://modelo');
        console.log('Se cargo modelo');
    }

    if (cmodel == null) {
        cmodel = tf.sequential();
        const hidden = tf.layers.dense({
            units: 6,
            inputShape: [2],
            activation: 'relu'
        });
        const output = tf.layers.dense({
            units: 1,
        });
        cmodel.add(hidden);
        cmodel.add(output);
        console.log('Creando modelo');
    }
    cmodel.compile({
        optimizer: tf.train.sgd(0.001),
        loss: 'meanSquaredError'
    });
    return cmodel;

}

// End ---

// Functions for buttons
async function btnTrain() {
    await train(model);
    console.log('Training Complete');

}

function btnCalculate(a, b) {
    let outputs = model.predict(tf.tensor2d([a, b], [1, 2]));
    document.getElementById('result').innerHTML = outputs.toString();
    outputs.print();
}

// End ---