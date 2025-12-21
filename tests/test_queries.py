from pipeline.run import process


def test_active_providers_ca():
    res = process('Find active providers in California hospitals', schema_path='schema/schema.json')
    q = res['query']
    assert 'providers' in q
    assert 'affiliations' in q
    assert 'status' in q
    assert '"ACTIVE"' in q or '"CA"' in q
