# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy as np
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ColorField,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation("color_cast_correction", title="Color Cast Correction", tags=["color", "correction"], version="1.0.0")
class ColorCastCorrectionInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Correct color cast while preserving perceptual brightness"""

    image: ImageField = InputField(description="Image to correct")
    strength: float = InputField(default=0.75, ge=0, le=2, description="Strength of correction adjustment")
    target_color: ColorField = InputField(
        default=ColorField(r=127, g=127, b=127, a=255), description="The target color for correction"
    )

    def srgb_to_linear(self, rgb):
        a = 0.055
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    def linear_to_srgb(self, linear):
        a = 0.055
        linear = np.clip(linear, 0.0, None)
        return np.where(linear <= 0.0031308, 12.92 * linear, (1 + a) * linear ** (1 / 2.4) - a)

    def rgb_to_oklch(self, rgb):
        rgb = np.clip(rgb, 0.0, 1.0)
        linear = self.srgb_to_linear(rgb)

        M = np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )
        lms = linear @ M.T
        lms = np.cbrt(lms)

        lab = np.tensordot(
            lms,
            np.array(
                [
                    [0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660],
                ]
            ),
            axes=([2], [1]),
        )

        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.sqrt(a**2 + b**2)
        H = np.arctan2(b, a)
        H = np.degrees(H) % 360.0

        return np.stack([L, C, H], axis=-1)

    def oklch_to_rgb(self, oklch):
        L, C, H = oklch[..., 0], oklch[..., 1], oklch[..., 2]
        a = C * np.cos(np.radians(H))
        b = C * np.sin(np.radians(H))

        lab = np.stack([L, a, b], axis=-1)

        lms = np.tensordot(
            lab,
            np.array(
                [
                    [1.0000000000, 0.3963377774, 0.2158037573],
                    [1.0000000000, -0.1055613458, -0.0638541728],
                    [1.0000000000, -0.0894841775, -1.2914855480],
                ]
            ),
            axes=([2], [1]),
        )
        lms = lms**3

        linear = np.tensordot(
            lms,
            np.array(
                [
                    [4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010],
                ]
            ),
            axes=([2], [1]),
        )

        rgb = self.linear_to_srgb(linear)
        return np.clip(rgb, 0.0, 1.0)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        img_np = np.asarray(image).astype(np.float32) / 255.0
        if img_np.shape[2] == 4:
            alpha = img_np[..., 3:]
            img_np = img_np[..., :3]
        else:
            alpha = None

        height, width, _ = img_np.shape

        oklch_orig = self.rgb_to_oklch(img_np)
        original_L = oklch_orig[..., 0]

        target_rgb = np.array([self.target_color.r, self.target_color.g, self.target_color.b], dtype=np.float32) / 255.0
        mean_rgb = img_np.mean(axis=(0, 1))
        gain = target_rgb / (mean_rgb + 1e-6)
        gain = 1.0 + self.strength * (gain - 1.0)
        r_gain, g_gain, b_gain = gain

        corrected = img_np.copy()
        corrected[..., 0] *= r_gain
        corrected[..., 1] *= g_gain
        corrected[..., 2] *= b_gain
        corrected = np.clip(corrected, 0.0, 1.0)

        oklch_corr = self.rgb_to_oklch(corrected)
        oklch_corr[..., 0] = original_L
        final_rgb = self.oklch_to_rgb(oklch_corr)
        if alpha is not None:
            final_rgb = np.concatenate([final_rgb, alpha], axis=-1)

        output_image = Image.fromarray((final_rgb * 255).astype(np.uint8))
        image_dto = context.images.save(image=output_image)
        return ImageOutput.build(image_dto)
