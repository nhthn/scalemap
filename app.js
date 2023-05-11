const synth = new Tone.Synth().toDestination();

const root = 260.0;

let sequence = null;

Tone.Transport.start();

async function playScale(scale, edo) {
    await Tone.start();
    if (sequence !== null) {
        sequence.stop();
    }
    sequence = new Tone.Sequence((time, note) => {
        synth.triggerAttackRelease(root * Math.pow(2, note / edo), 0.3);
    }, scale.concat([scale[0] + edo]));
    sequence.loop = false;
    sequence.start();
}