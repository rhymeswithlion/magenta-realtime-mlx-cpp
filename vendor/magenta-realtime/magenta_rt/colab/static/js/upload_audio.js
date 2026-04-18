
/**
 * Converts a buffer to a base64 string.
 * @param {!ArrayBuffer} buffer The buffer to convert.
 * @return {!Promise<string>} The base64 string.
 */
async function bufferToBase64(buffer) {
  const base64url = await new Promise((r) => {
    const reader = new FileReader();
    reader.onload = () => r(reader.result);
    reader.readAsDataURL(new Blob([buffer]));
  });
  return base64url.slice(37);
}

/**
 * Opens a file upload dialog and sends the selected audio data to the backend.
 * @param {string} callbackName The name of the callback function to invoke.
 * @return {!Promise<null>} A promise that resolves when the audio data has been
 *     sent or when the user cancels the file selection.
 */
function uploadAudio(callbackName) {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.onchange = () => {
      const file = input.files[0];

      if (!file) {
        console.log('User cancelled file selection.');
        resolve(null);
        return;
      }

      const reader = new FileReader();

      reader.onload = (ev) => {
        window.audioContext.decodeAudioData(reader.result)
            .then((decoded) => {
              let buffer = decoded.getChannelData(0);
              buffer = buffer.slice(0, 10 * window.sampleRate);
              bufferToBase64(buffer).then((buffer) => {
                google.colab.kernel.invokeFunction(
                    callbackName, [file.name, buffer, window.sampleRate], {});
              });
            })
            .catch(() => {
              alert('Could not decode audio data');
            });
        resolve(null);
      };

      reader.readAsArrayBuffer(file);
    };

    input.click();
  });
}
