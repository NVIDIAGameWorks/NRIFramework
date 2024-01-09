#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef _MSC_VER
#include <strings.h>
#endif

#include "detex.h"
#include "misc.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

// Load texture from an image file using stb_image (first mip-map only).
// Formats supported: JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, PNM.
// Returns true if successful. The texture is allocated, free with free().
bool detexLoadImageFile(const char *filename, detexTexture ***texture_out) {
	int x, y, comp;
	unsigned char *image = stbi_load(filename, &x, &y, &comp, STBI_rgb_alpha);
	if (!image)
	{
		detexSetErrorMessage("detexLoadImageFile: Could not open file %s. Reason: %s",
			filename, stbi_failure_reason());
		return false;
	}
	detexTexture **dTexture = (detexTexture **)malloc(sizeof(detexTexture *));
	dTexture[0] = (detexTexture *)malloc(sizeof(detexTexture));
	dTexture[0]->format = DETEX_PIXEL_FORMAT_RGBA8;
	dTexture[0]->width = x;
	dTexture[0]->height = y;
	dTexture[0]->width_in_blocks = x;
	dTexture[0]->height_in_blocks = y;
	size_t size = x * y * detexGetPixelSize(DETEX_PIXEL_FORMAT_RGBA8);
	dTexture[0]->data = (uint8_t *)malloc(size);
	memcpy(dTexture[0]->data, image, size);
	*texture_out = dTexture;
	stbi_image_free(image);
	return true;
}

// Save texture to an image file (type autodetected from extension).
// Formats supported: JPEG, PNG, TGA, BMP.
DETEX_API bool detexSaveImageFile(detexTexture *texture, const char *filename)
{
	if (texture->format != DETEX_PIXEL_FORMAT_RGBA8)
	{
		detexSetErrorMessage("detexSaveImageFile: Only RGA8 format supported");
		return false;
	}
	size_t filename_length = strlen(filename);
	if ((filename_length > 4 && strncasecmp(filename + filename_length - 4, ".jpg", 4) == 0) ||
		(filename_length > 5 && strncasecmp(filename + filename_length - 5, ".jpeg", 5) == 0))
		return stbi_write_jpg(filename, texture->width, texture->height, 4, texture->data, 0);
	else if (filename_length > 4 && strncasecmp(filename + filename_length - 4, ".png", 4) == 0)
		return stbi_write_png(filename, texture->width, texture->height, 4, texture->data, 4 * texture->width);
	else if (filename_length > 4 && strncasecmp(filename + filename_length - 4, ".tga", 4) == 0)
		return stbi_write_tga(filename, texture->width, texture->height, 4, texture->data);
	else if (filename_length > 4 && strncasecmp(filename + filename_length - 4, ".bmp", 4) == 0)
		return stbi_write_bmp(filename, texture->width, texture->height, 4, texture->data);
	else {
		detexSetErrorMessage("detexSaveImageFile: Unsupported filename extension");
		return false;
	}
}
