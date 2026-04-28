import pandas as pd

# 1. Define the descriptions for your 10 specific tags
tag_descriptions = {
    "java": "A high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible.",
    "html": "HyperText Markup Language, the standard markup language for documents designed to be displayed in a web browser.",
    "javascript": "A lightweight, interpreted programming language with first-class functions, core to interactive web development.",
    "jquery": "A fast, small, and feature-rich JavaScript library designed to simplify HTML DOM tree traversal and manipulation.",
    "c++": "A general-purpose programming language created as an extension of C, known for performance, efficiency, and hardware control.",
    "c#": "A modern, object-oriented, and type-safe programming language developed by Microsoft for the .NET framework.",
    "php": "A popular general-purpose scripting language that is especially suited to server-side web development.",
    "python": "An interpreted, high-level programming language known for its code readability and comprehensive standard library.",
    "android": "A mobile operating system based on a modified version of the Linux kernel, designed primarily for touchscreen mobile devices.",
    "ios": "A mobile operating system created and developed by Apple Inc. exclusively for its hardware like the iPhone and iPad."
}

def add_descriptions():
    # 2. Load the original CSV
    file_path = 'data/Table_B_Tags_Micro.csv'
    df = pd.read_csv(file_path)

    # 3. Create the new 'Description' column by mapping the dictionary to the 'concept_name' column
    df['Description'] = df['concept_name'].map(tag_descriptions)

    # 4. Save the updated dataframe to a new CSV
    output_path = 'table_b_stack.csv'
    df.to_csv(output_path, index=False)

    print(f"Successfully added descriptions to {len(df)} tags.")
    print(f"Saved output to: {output_path}\n")
    print(df.head())

if __name__ == "__main__":
    add_descriptions()