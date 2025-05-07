from string import Template
from textwrap import dedent


class HTMLTemplate:
    """HTML hack for styling."""

    base_style = Template(
        dedent(
            """
            <style>
                $css
            </style>"""
        )
    )


class CSS:
    """Define CSS styles."""

    page_style = """
    .st-key-app_css button {
        border-radius: 25px;
        box-shadow: 3px 5px 10px 0px rgba(128, 128, 128, 0.245);
        position: fixed;
        top: 4rem;
        right: 2rem;
    }
    details {
        color: grey;
    }
    summary {
        color: grey;
        font-weight: bold;
    }
    """
