"""
===============================================================================
UNIT TEST SUITE: test_nlp_locators.py
LOCATION:        tests/test_nlp_locators.py
OBJECTIVE:       Unit testing suite for NLP prompt locator parsing and 
                 3D spatial ROI mask generation (scripts/phase_2a_rule_based/nlp_locators.py).
===============================================================================
"""

import sys
import unittest
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common.nlp_locators import (
    parse_prompt_spatial_locators,
    generate_text_spatial_mask,
    ANATOMICAL_LOCATOR_KEYWORDS,
)



class TestNLPLocators(unittest.TestCase):

    def test_parse_prompt_spatial_locators_basic(self):
        """Test extraction of basic directional keywords ('right lower lobe')."""
        bounds = parse_prompt_spatial_locators("Ground-glass opacity in the right lower lobe")
        self.assertIn("rl", bounds)
        self.assertEqual(bounds["rl"], [0.45, 1.0])
        self.assertIn("is", bounds)
        self.assertEqual(bounds["is"], [0.0, 0.55])

    def test_parse_prompt_spatial_locators_apical(self):
        """Test extraction of apical/upper lobe keywords ('apical emphysema left lung')."""
        bounds = parse_prompt_spatial_locators("Apical emphysema in left lung")
        self.assertIn("rl", bounds)
        self.assertEqual(bounds["rl"], [0.0, 0.55])
        self.assertIn("is", bounds)
        self.assertEqual(bounds["is"], [0.60, 1.0])

    def test_generate_text_spatial_mask_shape(self):
        """Test shape and value bounds of generated 3D text ROI mask."""
        shape = (32, 32, 32)
        mask = generate_text_spatial_mask("Atelectasis in the left lower lobe", shape)
        self.assertEqual(mask.shape, shape)
        self.assertEqual(mask.dtype, np.float32)
        # Right side (X >= 18) should be 0.0 for 'left'
        self.assertTrue(np.all(mask[18:, :, :] == 0.0))

    def test_empty_prompt_fallback(self):
        """Test fallback to full ones mask for empty or non-string prompts."""
        shape = (16, 16, 16)
        mask_empty = generate_text_spatial_mask("", shape)
        self.assertTrue(np.all(mask_empty == 1.0))

        bounds_none = parse_prompt_spatial_locators(None)
        self.assertEqual(bounds_none, {})


if __name__ == "__main__":
    unittest.main()
