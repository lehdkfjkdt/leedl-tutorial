import fitz, os

pdf = r"d:\Dtpmh_Accumulate\dtpmh_accumulate\Professional and Academic Accumulation\learning\paper-review\π0.7\pi07.pdf"
out = r"d:\Dtpmh_Accumulate\dtpmh_accumulate\Professional and Academic Accumulation\learning\paper-review\π0.7\pi07_paper_guide_figures"

doc = fitz.open(pdf)
# Fig.2=Page4, Fig.3=Page6, Fig.19=Page22 (0-indexed: 3,5,21)
for pg in [3, 5, 21]:
    page = doc[pg]
    for i, img in enumerate(page.get_images(full=True)):
        base = doc.extract_image(img[0])
        sz = len(base["image"])
        if sz > 5000:
            fn = f"fig_page{pg+1}_img{i+1}.{base['ext']}"
            path = os.path.join(out, fn)
            with open(path, "wb") as f:
                f.write(base["image"])
            print(f"Page {pg+1}, img {i+1}: {sz:>8}B -> {fn}")
doc.close()
print("done")
