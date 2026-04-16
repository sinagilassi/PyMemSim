# =======================================
# ! LOAD THERMODB INLINE SOURCE
# =======================================
# SECTION: reference content
# NOTE: used for eos
REFERENCE_CONTENT = """
REFERENCES:
    CUSTOM-REF-1:
      DATABOOK-ID: 1
      TABLES:
        ideal-gas-heat-capacity:
          TABLE-ID: 1
          DESCRIPTION:
            This table provides the ideal gas heat capacity (Cp_IG) in J/mol.K as a function of temperature (T) in K.
            Two correlation forms are used depending on the compound.
            Equation A.1 (dimensionless form Cp/R), Equation A.2 (absolute Cp form)
          EQUATIONS:
            EQ-1:
              BODY:
                - y = args['temperature | T | K'] / (parms['A | A | 1'] + args['temperature | T | K'])
                - cp_r = parms['B | B | 1'] + (parms['C | C | 1'] - parms['B | B | 1']) * math.pow(y,2) * (
                    1 + (y - 1) * (
                      parms['D | D | 1'] +
                      parms['E | E | 1'] * y +
                      parms['F | F | 1'] * math.pow(y,2) +
                      parms['G | G | 1'] * math.pow(y,3)
                    )
                  )
                - res['ideal-gas-heat-capacity | Cp_IG | J/mol.K'] = parms['universal-gas-constant | R | J/mol.K'] * cp_r
              BODY-INTEGRAL:
                None
              BODY-FIRST-DERIVATIVE:
                None
              BODY-SECOND-DERIVATIVE:
                None
            EQ-2:
              BODY:
                - term1 = parms['C | C | 1'] / args['temperature | T | K']
                - term2 = parms['E | E | 1'] / args['temperature | T | K']
                - res['ideal-gas-heat-capacity | Cp_IG | J/mol.K'] = (
                    parms['A | A | 1'] +
                    parms['B | B | 1'] * math.pow(term1 / math.sinh(term1), 2) +
                    parms['D | D | 1'] * math.pow(term2 / math.cosh(term2), 2)
                  ) / 1000.0
              BODY-INTEGRAL:
                None
              BODY-FIRST-DERIVATIVE:
                None
              BODY-SECOND-DERIVATIVE:
                None
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,A,B,C,D,E,F,G,universal-gas-constant,Eq]
            SYMBOL: [None,None,None,None,A,B,C,D,E,F,G,R,Cp_IG]
            UNIT: [None,None,None,None,1,1,1,1,1,1,1,J/mol.K,J/mol.K]
          VALUES:
            - [1,'water','H2O','g',33484.75,9275.30,1218.48,20241.42,2919.59,0,0,8.314,2]
            - [3,'hydrogen chloride','HCl','g',432.77152,3.42841,3.54436,-10.49775,-77.37691,403.95083,-442.34632,8.314,1]
        general-data:
          TABLE-ID: 2
          DESCRIPTION:
            This table provides general thermodynamic and physical data for selected compounds.
            Data includes critical properties, acentric factor, phase-change temperatures, and standard formation properties.
          DATA: []
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,critical-temperature,critical-pressure,critical-molar-volume,molecular-weight,acentric-factor,boiling-temperature,melting-temperature,enthalpy-of-fusion,enthalpy-of-formation,gibbs-energy-of-formation]
            SYMBOL: [None,None,None,None,Tc,Pc,Vc,MW,AcFa,Tb,Tm,EnFus,EnFo_IG,GiEnFo_IG]
            UNIT: [None,None,None,None,K,bar,cm3/mol,g/mol,None,K,K,J/g,J/mol,J/mol]
            CONVERSION: [None,None,None,None,1,1,1,1,1,1,1,1,1,1]
          VALUES:
            - [1,'water','H2O','l',647.096,220.640,55.947,18.015,0.3443,373.13,273.15,333.1,-241820,-228590]
            - [3,'hydrogen chloride','HCl','g',324.550,82.631,88.719,36.461,0.1280,188.20,158.95,54.9,-92310,-95300]
        vapor-pressure:
          TABLE-ID: 3
          DESCRIPTION:
            This table provides the vapor pressure (P) in bar as a function of temperature (T) in K.
            The correlation follows Table A.2. Critical pressure (Pc) is taken from Table A.1.
            Since Pc is stored in bar, the correlation first computes vapor pressure in bar, then converts to Pa.
          EQUATIONS:
            EQ-1:
              BODY:
                - Tr = args['temperature | T | K'] / parms['critical-temperature | Tc | K']
                - tau = 1 - Tr
                - expo = (1 / Tr) * (
                    parms['A | A | 1'] * tau +
                    parms['B | B | 1'] * math.pow(tau, 1.5) +
                    parms['C | C | 1'] * math.pow(tau, 2.5) +
                    parms['D | D | 1'] * math.pow(tau, 5)
                  )
                - ps_bar = parms['critical-pressure | Pc | bar'] * math.exp(expo)
                - res['vapor-pressure | VaPr | bar'] = ps_bar
              BODY-INTEGRAL:
                None
              BODY-FIRST-DERIVATIVE:
                None
              BODY-SECOND-DERIVATIVE:
                None
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,A,B,C,D,critical-temperature,critical-pressure,Eq]
            SYMBOL: [None,None,None,None,A,B,C,D,Tc,Pc,VaPr]
            UNIT: [None,None,None,None,1,1,1,1,K,bar,bar]
          VALUES:
            - [1,'water','H2O','g',-7.870154,1.906774,-2.310330,-2.063390,647.096,220.640,1]
            - [3,'hydrogen chloride','HCl','g',-6.454142,0.934797,-0.636477,-1.704349,324.550,82.631,1]
        liquid-heat-capacity:
          TABLE-ID: 4
          DESCRIPTION:
            This table provides the liquid heat capacity at constant pressure (Cp_LIQ) in J/mol.K as a function of temperature (T) in K.
            The correlation follows Table A.5.
            Critical temperature (Tc) and molecular weight (MW) are taken from Table A.1.
            The source verification values in the book are reported in J/(g.K), while this implementation returns J/mol.K.
          EQUATIONS:
            EQ-1:
              BODY:
                - tau = 1 - args['temperature | T | K'] / parms['critical-temperature | Tc | K']
                - res['liquid-heat-capacity | Cp_LIQ | J/mol.K'] = parms['universal-gas-constant | R | J/mol.K'] * (
                    parms['A | A | 1'] / tau +
                    parms['B | B | 1'] +
                    parms['C | C | 1'] * tau +
                    parms['D | D | 1'] * math.pow(tau, 2) +
                    parms['E | E | 1'] * math.pow(tau, 3) +
                    parms['F | F | 1'] * math.pow(tau, 4)
                  )
              BODY-INTEGRAL:
                None
              BODY-FIRST-DERIVATIVE:
                None
              BODY-SECOND-DERIVATIVE:
                None
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,A,B,C,D,E,F,critical-temperature,molecular-weight,universal-gas-constant,Eq]
            SYMBOL: [None,None,None,None,A,B,C,D,E,F,Tc,MW,R,Cp_LIQ]
            UNIT: [None,None,None,None,1,1,1,1,1,1,K,g/mol,J/mol.K,J/mol.K]
          VALUES:
            - [1,'water','H2O','l',0.255980,12.545950,-31.408960,97.766500,-145.423600,87.018500,647.096,18.015,8.314,1]
            - [3,'hydrogen chloride','HCl','l',0.428824,7.229828,-9.908417,35.977597,-73.966366,63.001991,324.550,36.461,8.314,1]
        enthalpy-of-vaporization:
          TABLE-ID: 5
          DESCRIPTION:
            This table provides the enthalpy of vaporization (EnVap) in J/mol as a function of temperature (T) in K.
          EQUATIONS:
            EQ-1:
              BODY:
                - t = 1 - args['temperature | T | K']/parms['critical-temperature | Tc | K']
                - parms['A | A | 1'] = parms['A | A | 1'] * math.pow(t, 1/3)
                - parms['B | B | 1'] = parms['B | B | 1'] * math.pow(t, 2/3)
                - parms['C | C | 1'] = parms['C | C | 1'] * math.pow(t, 1)
                - parms['D | D | 1'] = parms['D | D | 1'] * math.pow(t, 2)
                - parms['E | E | 1'] = parms['E | E | 1'] * math.pow(t, 6)
                - parms['F | F | 1'] = parms['universal-gas-constant | R | J/mol.K'] * parms['critical-temperature | Tc | K']
                - res['enthalpy-of-vaporization | EnVap | J/mol'] = parms['F | F | 1'] * (parms['A | A | 1'] + parms['B | B | 1'] + parms['C | C | 1'] + parms['D | D | 1'] + parms['E | E | 1'])
              BODY-INTEGRAL:
                  None
              BODY-FIRST-DERIVATIVE:
                  None
              BODY-SECOND-DERIVATIVE:
                  None
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,A,B,C,D,E,critical-temperature,universal-gas-constant,Eq]
            SYMBOL: [None,None,None,None,A,B,C,D,E,Tc,R,EnVap]
            UNIT: [None,None,None,None,1,1,1,1,1,K,J/mol.K,J/mol]
          VALUES:
            - [1,'water','H2O','l',6.853064,7.437940,-2.937398,-3.282184,8.396833,647.096,8.314,1]
            - [3,'hydrogen chloride','HCl','g',5.385594,3.577607,1.702220,-4.769082,5.095527,324.550,8.314,1]
        liquid-density:
          TABLE-ID: 6
          DESCRIPTION:
            This table provides liquid density (rho_LIQ) in kg/m3 as a function of temperature (T) in K.
            Most compounds use the Table A.3 coefficient form.
            Water is treated separately because Table A.3 refers to Eq. 3.52 instead of A, B, C, D coefficients.
            Critical temperature (Tc) is taken from Table A.1.
          EQUATIONS:
            EQ-1:
              BODY:
                - tau = 1 - args['temperature | T | K'] / parms['critical-temperature | Tc | K']
                - res['liquid-density | rho_LIQ | kg/m3'] = parms['critical-density | Dc | kg/m3'] + (
                    parms['A | A | 1'] * math.pow(tau, 0.35) +
                    parms['B | B | 1'] * math.pow(tau, 2/3) +
                    parms['C | C | 1'] * math.pow(tau, 1.0) +
                    parms['D | D | 1'] * math.pow(tau, 4/3)
                  )
              BODY-INTEGRAL:
                None
              BODY-FIRST-DERIVATIVE:
                None
              BODY-SECOND-DERIVATIVE:
                None
          STRUCTURE:
            COLUMNS: [No.,Name,Formula,State,A,B,C,D,critical-temperature,critical-density,Eq]
            SYMBOL: [None,None,None,None,A,B,C,D,Tc,Dc,rho_LIQ]
            UNIT: [None,None,None,None,1,1,1,1,K,kg/m3,kg/m3]
          VALUES:
            - [1,'water','H2O','l',0,0,0,0,647.096,322.00,1]
            - [3,'hydrogen chloride','HCl','l',981.8765,-441.4813,1121.7449,-553.3618,324.550,410.97,1]
"""
