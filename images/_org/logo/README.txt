The original logo uses a font (Nunito) that is not available on most clients, and we don't want to download that font
dynamically just to render the logo. Instead we convert the text to a path using the following approach:

- Download and install SkyFonts as described here:
  http://googledevelopers.blogspot.co.uk/2013/05/download-google-fonts-to-your-desktop.html
- Install the Nunito fonts.
- In Inkscape, use Path > Object to Path to convert the text to a path.

logo_org.svg is the original logo created by hand. logo_raw.svg is the logo with text imported into Inkscape (Inkscape
doesn't render the original logo correctly). ../../logo.svg is the modified file with text converted to path.