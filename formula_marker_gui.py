import base64
import io
import os
import re
from tkinter import Tk, Button, CENTER, Label, Text, END

import matplotlib.pyplot as plt
import pandas as pd
from IPython import InteractiveShell
from PIL import ImageTk, Image

from IPython.display import Math
import IPython

from IPython.testing.globalipapp import get_ipython
from svg2png import svg2png

shell = InteractiveShell.instance()

tags_to_remove = [
    r'\mathrm',
    r'\text',
    r'\textbf',
    r'\quad',
    r'\qquad',
    r'\textstyle',
    r'\displaystyle',
    r'<|im_end|>'
]


def remove_tag(s: str, tag: str) -> str:
    new_str = []

    i = 0
    while i < len(s):
        if s[i:i + len(tag)] == tag:
            i += len(tag) + 1  # +1 because of `{` after tag
            bracers_opened = 1
            while i < len(s) and (bracers_opened > 0 or s[i] != '}'):
                if s[i] == '{':
                    bracers_opened += 1
                if s[i] == '}':
                    bracers_opened -= 1
                    if bracers_opened == 0:
                        # remove last close bracket
                        continue
                new_str.append(s[i])
                i += 1

        if i >= len(s):
            break
        new_str.append(s[i])
        i += 1
    return ''.join(new_str)


def remove_custom_bracers(v):
    v = v.replace('\left(', '(')
    v = v.replace('\left[', '(')
    v = v.replace('\left{', '(')
    v = v.replace('\right)', '(')
    v = v.replace('\right]', ')')
    v = v.replace('\right}', ')')
    return v


def remove_extra_bracers_in_bottom_indices(v):
    pattern = re.compile('_(\{([^\s\\\}]*)\})?')
    v = pattern.sub('\\1', v)
    return v


# regexps = [re.compile(tag + '\s*(\{([^\}]*)\})?') for tag in tags_to_remove]

def filter_out_tags(v):
    for tag in tags_to_remove:
        v = remove_tag(v, tag)
        v = remove_custom_bracers(v)
        v = remove_extra_bracers_in_bottom_indices(v)
    return v


def render_latex(txt):
    # fig = plt.figure(figsize=(14, 3))
    # # fig.text(0, 0.6, txt.replace('$$', '$').replace('\\text', '\mathrm'), fontsize=20)
    # fig.Math(txt)
    #
    # # hide axes
    # gca = plt.gca()
    # gca.axes.axis('off')
    # # gca.axes.get_yaxis().set_visible(False)
    #
    # # Save PNG to memory buffer as BytesIO
    # from io import BytesIO
    # buffer = BytesIO()
    #
    # plt.savefig(buffer, format='JPEG')
    # plt.close()

    svg_data = shell.display_formatter.format(Math(txt), "image/svg+xml")[0]

    png_data = svg2png(svg_data)

    # Загружаем в PIL
    img = Image.open(io.BytesIO(png_data))

    return img#Image.open(buffer)


def to_image(base64_str: str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, 'utf-8'))))


INPUT_DIR = 'data_learn_output'
OUTPUT_DIR = 'formula_marker_output'
REMOVED_LABEL = '--REMOVED--'

if __name__ == '__main__':

    # load data
    dfs = {}
    for file in os.listdir(INPUT_DIR):
        df = pd.read_csv(f'{INPUT_DIR}/{file}', index_col='Unnamed: 0')
        if not df.empty:
            dfs[file] = df

    # Filter out blanks
    for k in dfs:
        df = dfs[k]
        df = df[df['prediction'].apply(lambda x: '$' in x)]
        df.loc[:, 'prediction'] = df['prediction'].apply(lambda x: re.sub('^[^\$]*\$', '$', x))
        df = df.reset_index().drop('index', axis=1)
        dfs[k] = df

    new_df = []
    new_dfs = {}

    # Filter out already processed entries
    for k in dfs:
        output_file = f'{OUTPUT_DIR}/{k}'
        if os.path.exists(output_file):
            # file exists, load processed entries
            df_processed = pd.read_csv(output_file)
            df_original = dfs[k]
            dfs[k] = df_original[~df_original['image'].isin(set(df_processed['image']))]


    def next_data() -> tuple[Image, Image, str]:
        global new_df, new_dfs

        for df_name, df in dfs.items():
            new_df = []
            new_dfs[df_name] = new_df

            df_len = df.shape[0]
            doc_label.config(text=f'Сейчас обрабатывается: {df_name}')
            for i, row in df.iterrows():
                image_text = row['image']
                image = to_image(image_text)
                prediction_text = row['prediction']

                image_ratio = image.size[0] / image.size[1]
                if image.size[0] > 1200:
                    new_width = int(1200)
                    new_height = int(1200 / image_ratio)
                    image = image.resize(size=(new_width, new_height))

                if image.size[1] > 200:
                    new_width = int(200 * image_ratio)
                    new_height = 200
                    image = image.resize(size=(new_width, new_height))

                try:
                    prediction_image = render_latex(prediction_text)
                except:
                    prediction_image = None

                info_label.config(text=f'{i} / {df_len - 1}')
                yield image_text, image, prediction_image, prediction_text


    root = Tk()
    root.geometry('1600x1000')
    root.title('Docs-processing создание датасета :)')  # заголовок

    black_bg = ImageTk.PhotoImage(Image.new("RGB", (600, 200), (0, 0, 0)))

    canvas_1 = Label(root, image=black_bg)
    canvas_1.image = black_bg
    canvas_1.place(x=100, y=20)

    canvas_2 = Label(root, image=black_bg, anchor=CENTER)
    canvas_2.image = black_bg
    canvas_2.place(x=100, y=220)

    data_it = next_data()

    image_text, image, prediction_image, prediction_text = None, None, None, None


    def go_to_next_entry():
        global image_text, image, prediction_image, prediction_text
        image_text, image, prediction_image, prediction_text = next(data_it)
        tk_image1 = ImageTk.PhotoImage(image)
        canvas_1['image'] = tk_image1
        canvas_1.image = tk_image1
        canvas_1.place(x=100, y=20)

        text_box.replace('1.0', END, prediction_text)

        if prediction_image:
            tk_image2 = ImageTk.PhotoImage(prediction_image)
        else:
            tk_image2 = black_bg
        canvas_2['image'] = tk_image2
        canvas_2.image = tk_image2
        canvas_2.place(x=100, y=220)


    def clicked_redraw_latex():
        prediction_text = text_box.get('1.0', END)

        try:
            prediction_image = render_latex(prediction_text)
        except:
            prediction_image = None

        if prediction_image:
            tk_image2 = ImageTk.PhotoImage(prediction_image)
        else:
            tk_image2 = black_bg
        canvas_2['image'] = tk_image2
        canvas_2.image = tk_image2
        canvas_2.place(x=100, y=220)


    def clicked_yes():
        corrected_text = text_box.get('1.0', END)
        new_df.append((image_text, corrected_text))
        go_to_next_entry()


    def clicked_no():
        new_df.append((image_text, REMOVED_LABEL))
        go_to_next_entry()


    def clicked_save():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for df_name, df in new_dfs.items():
            df_to_save = pd.DataFrame.from_records(df, columns=['image', 'prediction'])
            output_file = f'{OUTPUT_DIR}/{df_name}'
            if os.path.exists(output_file):
                df_old = pd.read_csv(output_file, index_col='Unnamed: 0')
                df_to_save = pd.concat([df_old, df_to_save])
            df_to_save.to_csv(f'formula_marker_output/{df_name}')


    text_box = Text(root)
    text_box.place(x=100, y=700, width=1400, height=80)

    btn0 = Button(root, text='Перерисовать LaTeX', command=clicked_redraw_latex)
    btn0.place(x=750, y=800)

    btn1 = Button(root, text='Подтвердить', command=clicked_yes)
    btn1.place(x=650, y=900)

    btn2 = Button(root, text='Выкинуть', command=clicked_no)
    btn2.place(x=850, y=900)

    btn4 = Button(root, text='Сохранить', command=clicked_save)
    btn4.place(x=1200, y=900)

    info_label = Label(root)
    info_label.place(x=775, y=900)

    doc_label = Label(root)
    doc_label.place(x=400, y=850)

    go_to_next_entry()

    # image.show()
    # tk_image1 = ImageTk.PhotoImage(image)
    # label = tk.Label(window, image=self.tk_image1)
    # label.pack()
    # tk_image1 = ImageTk.PhotoImage(image)

    root.mainloop()

    # save results
