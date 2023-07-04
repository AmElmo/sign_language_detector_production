
// import * as tf from "@tensorflow/tfjs"

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const translation = document.getElementById('prediction');

// LOAD MODEL HERE !
const model_sign_language = tf.loadLayersModel('https://storage.googleapis.com/sign-language-model_v3/model.json');
console.log(typeof model)

const list_poses = []

var dict_words = {
  0: "Headache 🧠",
  1: "Sore throat 😵",
  2: "Cough 😷",
  3: "Fever 🤒",
  4: "Stomach aches 😖",
  5: "Tired 😓",
  6: "Runny nose 🤧",
  7: "Nausea 🤢",
  8: "Trouble breathing 🫁"
};

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.segmentationMask, 0, 0,
                      canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  // canvasCtx.globalCompositeOperation = 'source-in';
  canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  // canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);

  // canvasCtx.globalCompositeOperation = 'source-over';
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
                 {color: '#e545ed', lineWidth: 4});
  drawLandmarks(canvasCtx, results.poseLandmarks,
                {color: '#e545ed', lineWidth: 2});


  const pose = []
  if (results.poseLandmarks) {
      results.poseLandmarks.forEach((res) => {
        pose.push(res.x, res.y, res.z, res.visibility)
      })
    } else {
      for (let i=0; i<132; ++i) pose[i] = 0;
    }
  if (results.faceLandmarks) {
      results.faceLandmarks.forEach((res) => {
        pose.push(res.x, res.y, res.z)
    })} else {
      for (let i=132; i<1536; ++i) pose[i] = 0;
    }
  if (results.leftHandLandmarks) {
        results.leftHandLandmarks.forEach((res) => {
          pose.push(res.x, res.y, res.z)
      })} else {
        for (let i=1536; i<1599; ++i) pose[i] = 0;
      }
  if (results.rightHandLandmarks) {
        results.rightHandLandmarks.forEach((res) => {
          pose.push(res.x, res.y, res.z)
      })} else {
        for (let i=1599; i<1662; ++i) pose[i] = 0;
      }

  list_poses.push(pose)

  console.log(list_poses.length)

  if (list_poses.length > 29) {
    console.log(list_poses.length)
    console.log(JSON.stringify(list_poses))
    console.log(list_poses)

    const y = tf.tensor2d(list_poses);
    const axs = 0;
    const list_poses_3D = y.expandDims(axs)

    // Specific reshaping for graph model
    console.log(list_poses_3D.shape)

    console.log("Pushing to model!!!")

    // Define a variable to track whether a match was found
    let foundMatch = false;

    async function processModel(){
      const model = await tf.loadLayersModel('https://storage.googleapis.com/sign-language-model_v3/model.json');

      const prediction_1 = model.predict(list_poses_3D);
      console.log(prediction_1)

      // model.predict(list_poses_3D).print()

      // const predictions = await model.executeAsync(list_poses_3D, ['StatefulPartitionedCall/sequential_2/dense_3/BiasAdd/ReadVariableOp']).data()
      // console.log(predictions)

      const preds = prediction_1.dataSync();
      console.log(preds)

      for(const [i,v] of preds.entries()) {
        console.log(i,v)
        if (v > 0.80) {
          console.log(dict_words[i])
          translation.innerHTML = dict_words[i]
          foundMatch = true;
          break; // Exit the loop after finding a match, if desired
        }
    }
  }

  // If no match is found, set a default value
  if (foundMatch === false) {
    translation.innerHTML = "Make some signs 👋";
  }

    processModel()

    list_poses.length = 0
    list_poses_3D.length = 0
}


  drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
                 {color: '#36ebf5', lineWidth: 1});

  drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
                 {color: '#1c7d8a', lineWidth: 5});
  drawLandmarks(canvasCtx, results.leftHandLandmarks,
                {color: '#00FF00', lineWidth: 2});
  drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
                 {color: '#1c7d8a', lineWidth: 5});
  drawLandmarks(canvasCtx, results.rightHandLandmarks,
                {color: '#00FF00', lineWidth: 2});
  canvasCtx.restore();
}

const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});
holistic.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
