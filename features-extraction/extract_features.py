#import features_extraction as fe
#hiphop = []
#k = 0
#for i in y[1][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    hiphop.append(fe.feature_extraction(i))
#indie = []
#k = 0
#for i in y[2][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    indie.append(fe.feature_extraction(i))
#jazz = []
#k = 0
#for i in y[3][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    jazz.append(fe.feature_extraction(i))
#pop = []
#k = 0
#for i in y[4][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    pop.append(fe.feature_extraction(i))
#rock = []
#k = 0
#for i in y[5][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    rock.append(fe.feature_extraction(i))
#classical = []
#k = 0
#for i in y[6][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    classical.append(fe.feature_extraction(i))
#country = []
#k = 0
#for i in y[7][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    country.append(fe.feature_extraction(i))
#folk = []
#k = 0
#for i in y[8][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    folk.append(fe.feature_extraction(i))
#lounge = []
#k = 0
#for i in y[9][1:]:
#    k += 1
#    print(str(k) + ' / ' + str(len(y[0][1:])))
#    lounge.append(fe.feature_extraction(i))
    
#label = ['tempo', 'length', 'energy', 'energy ratio', 'main pitch', 'main interval 1', 'main interval 2', 'main pitch ratio']
#leg = ['classical', 'country', 'electro', 'folk', 'indie', 'jazz', 'lounge', 'pop', 'rock']
#for i in range(1,8):
#    for j in range(i):
#
#        plt.figure()
#        k = classical
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = country
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = electro
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = folk
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        #k = hiphop
#        #plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = indie
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = jazz
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = lounge
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = pop
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        k = rock
#        plt.plot([k[n][i] for n in range(len(k))],[k[n][j] for n in range(len(k))], 'o')
#        plt.xlabel(label[i])
#        plt.ylabel(label[j])
#        plt.legend(leg)