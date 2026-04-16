REFERENCE = """
cpig-cpliq-section-c:
  TABLE-ID: C-CPIG-4COMP
  DESCRIPTION:
    Section C: Ideal-gas heat capacity correlation and CpIG/Cpliq at 298.15 K for selected components.
    Source form: Cp^0/R = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
  EQUATIONS:
    EQ-1:
      BODY:
        - parms['a0 | a0 | 1'] = parms['a0 | a0 | 1']/1
        - parms['a1 | a1 | 1E3'] = parms['a1 | a1 | 1E3']/1
        - parms['a2 | a2 | 1E5'] = parms['a2 | a2 | 1E5']/1
        - parms['a3 | a3 | 1E8'] = parms['a3 | a3 | 1E8']/1
        - parms['a4 | a4 | 1E11'] = parms['a4 | a4 | 1E11']/1
        - res['dimensionless-heat-capacity | Cp0_R | 1'] = parms['a0 | a0 | 1'] + (parms['a1 | a1 | 1E3']/1.0E3)*args['temperature | T | K'] + (parms['a2 | a2 | 1E5']/1.0E5)*(args['temperature | T | K']**2) + (parms['a3 | a3 | 1E8']/1.0E8)*(args['temperature | T | K']**3) + (parms['a4 | a4 | 1E11']/1.0E11)*(args['temperature | T | K']**4)
        - res['ideal-gas-heat-capacity | CpIG | J/mol.K'] = 8.314462618 * res['dimensionless-heat-capacity | Cp0_R | 1']
      BODY-INTEGRAL:
        None
      BODY-FIRST-DERIVATIVE:
        None
      BODY-SECOND-DERIVATIVE:
        None
  STRUCTURE:
    COLUMNS: [No., Formula, Name, CAS #, Tmin, Tmax, a0, a1, a2, a3, a4, CpIG_298.15K, Cpliq_298.15K]
    SYMBOL: [No, Formula, Name, CAS, T_min, T_max, a0, a1, a2, a3, a4, CpIG_298, Cpliq_298]
    UNIT: [None, None, None, None, K, K, 1, 1E3, 1E5, 1E8, 1E11, J/mol.K, J/mol.K]
  VALUES:
    - [60, C2H4O2, acetic acid, 64-19-7, 50, 1000, 4.375, -2.397, 6.757, -8.764, 3.478, 63.44, 123.10]
    - [27, CH4O, methanol, 67-56-1, 50, 1000, 4.714, -6.986, 4.211, -4.443, 1.535, 44.06, 81.08]
    - [440, H2O, water, 7732-18-5, 50, 1000, 4.395, -4.186, 1.405, -1.564, 0.632, 33.58, 75.29]
    - [91, C3H6O2, methyl ethanoate (methyl acetate), 79-20-9, 298, 1000, 4.242, 14.388, 3.338, -4.930, 1.931, 85.30, 143.90]

"""
