import tempfile
import unittest
from pathlib import Path
import wave
import numpy as np
from analyze import demo, features, read_wav

class SignalTests(unittest.TestCase):
    def test_tone_and_silence(self):
        signal, rate = demo()
        rows = features(signal, rate)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]['rms_input'], .5 / np.sqrt(2), places=10)
        self.assertAlmostEqual(rows[0]['rms_normalized'], 1 / np.sqrt(2), places=10)
        self.assertEqual(rows[-1]['rms_normalized'], 0)
    def test_stereo_and_tail(self):
        rows = features(np.column_stack([np.ones(11), np.zeros(11)]), 4, 1, .5)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[3]['end_seconds'], 2.5)
        self.assertEqual(rows[0]['peak_normalized'], 1)
        self.assertEqual(rows[-1]['peak_normalized'], 0)
    def test_pcm_decoding(self):
        for width, dtype, values, expected in [
            (1, 'u1', [0,128,255], [-1,0,127/128]),
            (2, '<i2', [-32768,0,32767], [-1,0,32767/32768]),
            (4, '<i4', [-2147483648,0,2147483647], [-1,0,2147483647/2147483648])]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'sample.wav'
                with wave.open(str(path),'wb') as w:
                    w.setnchannels(1); w.setsampwidth(width); w.setframerate(8000)
                    w.writeframes(np.array(values,dtype=dtype).tobytes())
                data, rate = read_wav(path)
                np.testing.assert_allclose(data[:,0],expected)
                self.assertEqual(rate,8000)
    def test_invalid_inputs(self):
        for signal, rate, window, hop in [([],8,1,.5),([np.nan]*8,8,1,.5),([1]*8,0,1,.5),([1]*8,8,0,.5),([1]*8,8,1,0),([1]*4,8,1,.5)]:
            with self.assertRaises(ValueError): features(signal,rate,window,hop)
    def test_sign_changes(self):
        row=features([1,-1,1,-1],4,1,1)[0]
        self.assertEqual(row['zero_crossing_rate'],1)

if __name__ == '__main__': unittest.main()
