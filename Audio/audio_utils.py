import collections
import contextlib
import librosa
import numpy as np
import scipy
import os
import torch
import torchaudio
import wave
import webrtcvad


def read_wave(path):
    """ Reads a .wav file. """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "audio channel should be 1 (mono)"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "audio sample width should be 2 bytes"
        original_sample_rate = wf.getframerate()
        if original_sample_rate == 44100:
            sample_rate = 48000
        else:
            sample_rate = original_sample_rate
        assert sample_rate in (8000, 16000, 32000, 48000), "audio sample rate should be in [8000, 16000, 32000, 44100, 48000] Hz"
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        duration = frames / original_sample_rate
        return pcm_data, sample_rate, original_sample_rate, duration


def write_wave(path, audio, sample_rate):
    """ Writes a .wav file. """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def load_audio(path):
    """ Read the WAV audio file using torchaudio library """
    sound, sample_rate = torchaudio.load(path)
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound.numpy()


def video_to_audio(file_name):
    """ Transforms video file into a WAV audio file """
    try:
        file, extension = os.path.splitext(file_name)
        # Convert video into .wav file
        os.system('ffmpeg -y -i {file}{ext} -ac 1 -ar 48000 -acodec pcm_s16le {file}.wav'.format(file=file, ext=extension))
        print('"{}" successfully converted into WAV audio!'.format(file_name))
        return file+'.wav'
    except OSError as err:
        print(err.reason)
        exit(1)


def combine_audio(video_file_name, audio_file_name, final_video_file_name):
    """ Combine audio with a video file """
    try:
        # Convert video into .wav file
        os.system(f'ffmpeg -y -i {video_file_name} -i {audio_file_name}  -vcodec copy -acodec copy {final_video_file_name}')
        print('"{}" success!'.format(final_video_file_name))
    except OSError as err:
        print(err.reason)
        exit(1)


def load_decoder(labels):
    """ Utility function to create GreedyDecoder instance """
    decoder = GreedyDecoder(labels=labels,blank_index=labels.index('_'))
    return decoder


def smaller_length(length,input):
    """ Returns the spectrogram input that is converted to fixed sixe depending on the onnx model. """
    temp = np.zeros((161,256),dtype=float)
    for j in range(161):
        for k in range(1,length):
            temp[j,k]=input[j,k]
    temp = torch.FloatTensor(temp)
    temp = temp.view(1,1,temp.size(0),temp.size(1))
    input_array = temp.numpy()
    return input_array


class Frame(object):
    """ Represents a "frame" of audio data. """
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, original_sample_rate, frame_duration_ms, vad, frames, fps):
    """
    Filters out non-voiced audio frames.
    """
    correction_factor = 1
    correction_factor = sample_rate/original_sample_rate

    for frame_no, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        timestamp = correction_factor * frame_no * (frame_duration_ms / 1000.0) # * (30 / fps)
        yield int(timestamp*fps), frame, is_speech, timestamp


def vad_segment_generator(wavFile, aggressiveness, fps):
    """ Generate VAD segments. Filters out non-voiced audio frames. """
    audio, sample_rate, original_sample_rate, audio_length = read_wave(wavFile)
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, original_sample_rate, 30, vad, frames, int(fps))

    return segments, sample_rate, original_sample_rate, audio_length


class AudioParser(object):
    """ Base class for AudioParser """
    def parse_audio(self, audio_path):
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    """ SpectrogramParser - parses audio file into Spectogram """
    def __init__(self, sample_rate, normalize = True):
        super(SpectrogramParser, self).__init__()
        self.window_stride = 0.01
        self.window_size = 0.02
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.win_length = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)

    def parse_audio(self, audio_data=None, audio_path=None):
        """ Parses audio file into spectrogram with optional normalization and various augmentations """
        if audio_data is not None:
            y = audio_data
        elif audio_path is not None:
            y = load_audio(audio_path)
        else:
            return

        n_fft = self.win_length
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=self.hop_length,
                         win_length=self.win_length, window=scipy.signal.windows.hamming)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    def get_hop_length(self):
        """ Getter method for hop_length property """
        return self.hop_length

    def get_win_length(self):
        """ Getter method for win_length property """
        return self.win_length


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.
    """
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    """ GreedyDecoder class for decoding the Deepspeech output """
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes = None, remove_repetitions = False, return_offsets = False):
        """ Given a list of numeric sequences, returns the corresponding strings """
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        """ Utility function to process Deepspeech output string """

        string = ''
        offsets = []

        for i in range(size):
            char = self.int_to_char[sequence[i].item()]

            if char != self.int_to_char[self.blank_index]:
                # If this char is a repetition and remove_repetitions = true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix.
        Removes repeated elements in the sequence, as well as blanks.
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets
