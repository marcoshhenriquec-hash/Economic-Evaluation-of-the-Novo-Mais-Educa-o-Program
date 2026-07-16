import argparse
import csv
import re
import sys
from pathlib import Path
 
from pypdf import PdfReader
 
 
COLUMNS = [
    "pagina",
    "sg_uf",
    "no_municipio",
    "cnpj_executora",
    "executora",
    "sg_destinacao",
    "co_escola",
    "no_escola",
    "qt_alunos",
    "vl_custeio",
    "vl_capital",
    "vl_total",
    "esfera",
    "localizacao",
]
 
START_RE = re.compile(
    r"^(?P<uf>[A-Z]{2})\s*(?P<municipio>.*?)(?P<cnpj>\d{14})(?P<rest>.*)$"
)
 
DESTINATION_RE = re.compile(
    r"(?P<dest>"
    r"EDUC\.INT\.\s*P1\s*2016-DJ|"
    r"EDUC\.INT\.\s*1ªP\s*2018|"
    r"Ed\.Int\.\s*1ª\s*P\s*ref\s*2016|"
    r"Ed\.Int\.\s*2ª\s*P\s*2016|"
    r"Educ\.Int\.\s*2ªP\s*C?\s*2018"
    r")\s*(?P<co_escola>\d{8})(?P<after>.*)$",
    re.IGNORECASE,
)
 
END_RE = re.compile(r"(?P<esfera>MUNICIPAL|ESTADUAL)(?P<localizacao>URBANA|RURAL)\s*$")
 
 
def clean_text(value):
    return re.sub(r"\s+", " ", value).strip()
 
 
def parse_school_and_values(text):
    end_match = END_RE.search(text)
    if not end_match:
        raise ValueError("nao encontrou esfera/localizacao no fim da linha")
 
    values_text = text[: end_match.start()].strip()
    parts = values_text.rsplit(None, 3)
    if len(parts) != 4:
        raise ValueError("nao encontrou os campos de valores")
 
    school_and_students, vl_custeio, vl_capital, vl_total = parts
    school_match = re.match(r"^(?P<school>.*?)(?P<students>\d+)$", school_and_students)
    if not school_match:
        raise ValueError("nao separou nome da escola e quantidade de alunos")
 
    return {
        "no_escola": clean_text(school_match.group("school")),
        "qt_alunos": school_match.group("students"),
        "vl_custeio": vl_custeio,
        "vl_capital": vl_capital,
        "vl_total": vl_total,
        "esfera": end_match.group("esfera"),
        "localizacao": end_match.group("localizacao"),
    }
 
 
def parse_line(line, page_number):
    line = clean_text(line)
    start_match = START_RE.match(line)
    if not start_match:
        return None
 
    dest_match = DESTINATION_RE.search(start_match.group("rest"))
    if not dest_match:
        raise ValueError("nao encontrou destinacao/codigo da escola")
 
    school_values = parse_school_and_values(dest_match.group("after"))
 
    return {
        "pagina": page_number,
        "sg_uf": start_match.group("uf"),
        "no_municipio": clean_text(start_match.group("municipio")),
        "cnpj_executora": start_match.group("cnpj"),
        "executora": clean_text(start_match.group("rest")[: dest_match.start()]),
        "sg_destinacao": clean_text(dest_match.group("dest")),
        "co_escola": dest_match.group("co_escola"),
        **school_values,
    }
 
 
def find_default_pdf():
    pdfs = sorted(Path(".").glob("*.pdf"))
    if len(pdfs) == 1:
        return pdfs[0]
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado no diretorio atual.")
    raise SystemExit("Informe o PDF. Ha mais de um arquivo .pdf no diretorio.")
 
 
def convert(pdf_path, output_path, max_pages=None, delimiter=";"):
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    limit = min(max_pages or total_pages, total_pages)
 
    rows = []
    errors = []
    for page_index in range(limit):
        text = reader.pages[page_index].extract_text() or ""
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = parse_line(line, page_index + 1)
            except ValueError as exc:
                errors.append(
                    {
                        "pagina": page_index + 1,
                        "linha": line_number,
                        "erro": str(exc),
                        "texto": clean_text(line),
                    }
                )
                continue
            if row:
                rows.append(row)
 
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
 
    return total_pages, limit, len(rows), errors
 
 
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Converte o PDF de dados do SEI/FNDE em CSV."
    )
    parser.add_argument("pdf", nargs="?", type=Path, help="Arquivo PDF de entrada.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Arquivo CSV de saida. Padrao: mesmo nome do PDF com extensao .csv.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Processa apenas as N primeiras paginas do PDF.",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="Delimitador do CSV. Padrao: ';'.",
    )
    args = parser.parse_args(argv)
 
    pdf_path = args.pdf or find_default_pdf()
    output_path = args.output or pdf_path.with_suffix(".csv")
 
    total_pages, processed_pages, row_count, errors = convert(
        pdf_path=pdf_path,
        output_path=output_path,
        max_pages=args.max_pages,
        delimiter=args.delimiter,
    )
 
    print(f"PDF: {pdf_path}")
    print(f"Paginas processadas: {processed_pages} de {total_pages}")
    print(f"Linhas exportadas: {row_count}")
    print(f"CSV gerado: {output_path}")
 
    if errors:
        print(f"Avisos: {len(errors)} linhas pareciam dados, mas nao foram parseadas.")
        print("Primeiros avisos:")
        for error in errors[:10]:
            print(
                f"- pagina {error['pagina']}, linha {error['linha']}: "
                f"{error['erro']} | {error['texto'][:180]}"
            )
 
 
if __name__ == "__main__":
    main(sys.argv[1:])
